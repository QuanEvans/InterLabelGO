import os
from Bio import SeqIO
import argparse
from tqdm import tqdm
import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained
import esm
import numpy as np
import re
import json

from settings import settings_dict as settings

class PlmEmbed:

    def __init__(self,
        fasta_file:str,
        working_dir:str,
        model_name:str="esm2_t36_3B_UR50D",
        model_path:str=settings['esm3b_path'],
        use_gpu:bool=True,
        repr_layers:list=[34, 35, 36],
        include:list=["mean"],
        cache_dir:str=None,
    ):
        working_dir = os.path.abspath(working_dir)
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        if cache_dir is None:
            cache_dir = os.path.join(working_dir, "embed_feature")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.cache_dir = cache_dir

        self.use_gpu = use_gpu
        self.fasta_file = fasta_file
        # fasta for esm only contains the proteins that are not in the esm feature cache
        self.filltered_fasta_file = os.path.join(working_dir, f'filtered.fasta')
        self.repr_layers = repr_layers
        # note, if model_path is not provided, the model will be loaded from the path, however, the contact-regression should be put in same directory as the model_path
        self.model_path = model_path
        self.model_name = model_name
        self.include = include

    def parse_fasta(self, fasta_file=None)->dict:
        '''
        parse fasta file

        args:
            fasta_file: fasta file path
        return:
            fasta_dict: fasta dictionary {id: sequence}
        '''
        if fasta_file is None:
            fasta_file = self.fasta_file

        fasta_dict = {}
        for record in SeqIO.parse(fasta_file, 'fasta'):
            fasta_dict[record.id] = str(record.seq)
        return fasta_dict  
    
    def filter_fasta(self, fasta_file=None, cache_dir=None, filltered_fasta_file=None):
        """
        Only keep fasta sequences that are not in feature store directory

        Args:
            fasta_file (str, optional): fasta file path. Defaults to None.
        """
        if fasta_file is None:
            fasta_file = self.fasta_file
        if cache_dir is None:
            cache_dir = self.cache_dir
        if filltered_fasta_file is None:
            filltered_fasta_file = self.filltered_fasta_file

        fasta_dict = self.parse_fasta(fasta_file)
        # get processed fasta ids
        processed_ids = set()
        for file in os.listdir(cache_dir):
            if file.endswith('.npy'):
                processed_ids.add(file.split('.')[0])
        # filter fasta file
        filltered_fasta_dict = {k:v for k,v in fasta_dict.items() if k not in processed_ids}
        # write filtered fasta file
        with open(filltered_fasta_file, 'w') as f:
            for k,v in filltered_fasta_dict.items():
                f.write(f'>{k}\n{v}\n')
        return filltered_fasta_dict, filltered_fasta_file
    
    def extract(
            self,
            fasta_file:str=None, # path of fasta file to extract features from
            repr_layers:list = [34,35,36], # which layers to extract features from
            model_path:str = None, # path to model
            model_name:str = None, # name of model, if model_path is not provided
            use_gpu:bool = True, # use GPU if available
            truncate:bool = True, # truncate sequences longer than 1024 to match training setup
            include:list = ["mean", "per_tok", "bos", "contacts"], # which representations to return
            batch_size:int = 4096, # maximum batch size
            output_dir:str = None, # output directory for extracted representations
            overwrite:bool = False, # overwrite existing files
            model_type:str = "esm", # model type, esm or t5
            ) -> None:
        
        if model_type == "esm":
            self.esm_extract(
                fasta_file=fasta_file,
                repr_layers=repr_layers,
                model_path=model_path,
                model_name=model_name,
                use_gpu=use_gpu,
                truncate=truncate,
                include=include,
                batch_size=batch_size,
                output_dir=output_dir,
                overwrite=overwrite,
            )
        else:
            raise NotImplementedError(f"model type {model_type} is not implemented")
        
    def esm_extract(
            self,
            fasta_file:str=None, # path of fasta file to extract features from
            repr_layers:list = [34,35,36], # which layers to extract features from
            model_path:str = None, # path to model
            model_name:str = None, # name of model, if model_path is not provided
            use_gpu:bool = True, # use GPU if available
            truncate:bool = True, # truncate sequences longer than 1024 to match training setup
            include:list = ["mean", "per_tok", "bos", "contacts"], # which representations to return
            batch_size:int = 4096, # maximum batch size
            output_dir:str = None, # output directory for extracted representations
            overwrite:bool = False, # overwrite existing files
            ) -> None:
        
        if output_dir is None:
            output_dir = self.cache_dir
        if fasta_file is None:
            fasta_file = self.fasta_file     
        if model_name is None:
            model_name = self.model_name
        if model_path is None:
            model_path = self.model_path

        # filter fasta file
        if not overwrite:
            _, fasta_file = self.filter_fasta(fasta_file, output_dir)

        if os.path.getsize(fasta_file) == 0:
            # print function name
            print(f'### {self.extract.__name__} ###')
            print(f'All sequences are already processed in the default cache directory. Return.')
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if model_path and os.path.exists(model_path):
            model, alphabet = pretrained.load_model_and_alphabet(model_path)
        elif model_name:
            if model_name == "esm2_t33_650M_UR50D":
                model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            elif model_name == "esm2_t36_3B_UR50D":
                model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
            else:
                raise NotImplementedError(f"model {model_name} is not implemented")
        else:
            raise ValueError("model_path or model_name must be provided")
        model.eval()

        if torch.cuda.is_available() and use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            model = model.cuda()
            print("Transferred model to GPU")
        else:
            print("Using CPU")

        dataset = FastaBatchedDataset.from_file(fasta_file)
        batches = dataset.get_batch_indices(batch_size, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
        )

        assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
        repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]

        with torch.no_grad():
            for batch_idx, (lables, strs, toks) in tqdm(enumerate(data_loader), total=len(batches), desc="Extracting esm features", ascii=' >='):
                # print(
                #     f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
                # )
                if torch.cuda.is_available() and use_gpu:
                    toks = toks.to(device="cuda", non_blocking=True)

                # The model is trained on truncated sequences and passing longer ones in at
                # infernce will cause an error. See https://github.com/facebookresearch/esm/issues/21
                if truncate:
                    toks = toks[:, :1022]

                out = model(toks, repr_layers=repr_layers, return_contacts="contacts" in include)

                #logits = out["logits"].to(device="cpu")
                representations = {
                    layer: t.to(device="cpu") for layer, t in out["representations"].items()
                }
                if "contacts" in include:
                    contacts = out["contacts"].to(device="cpu")


                for i, label in enumerate(lables):

                    result = {"name": label}

                    if "per_tok" in include:
                        result["per_tok"] = {
                            layer: t[i, 1: len(strs[i]) + 1].clone().numpy()
                            for layer, t in representations.items()
                        }
                    
                    if "mean" in include:
                        result["mean"] = {
                        layer: t[i, 1: len(strs[i]) + 1].mean(0).clone().numpy()
                            for layer, t in representations.items()
                        }

                    if "bos" in include:
                        result["bos"] = {
                            layer: t[i, 0].clone() for layer, t in representations.items()
                        }

                    if "contacts" in include:
                        result["contacts"] = contacts[i, : len(strs[i]), : len(strs[i])].clone().numpy()
                    
                    if "sum" in include:
                        result["sum"] = {
                        layer: t[i, 1: len(strs[i]) + 1].sum(0).clone().numpy()
                            for layer, t in representations.items()
                        }
                    
                    if "max" in include:
                        result["max"] = {
                        layer: t[i, 1: len(strs[i]) + 1].max(0).values.cpu().numpy()
                            for layer, t in representations.items()
                        }
                    
                    if "min" in include:
                        result["min"] = {
                        layer: t[i, 1: len(strs[i]) + 1].min(0).values.cpu().numpy()
                            for layer, t in representations.items()
                        }

                    out_file = os.path.join(output_dir, f"{label}.npy")
                    np.save(out_file, result, allow_pickle=True)

                    if torch.cuda.is_available() and use_gpu:
                        torch.cuda.empty_cache()     





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta_file', type=str, help='path of fasta file')
    parser.add_argument('workdir', type=str, help='path of working directory')
    parser.add_argument('-mn', '--model_name', type=str, help='name of model', default="esm2_t36_3B_UR50D",)
    parser.add_argument('-mp', '--model_path', type=str, help='path of model', default=None)
    parser.add_argument('-c', '--cache_dir', type=str, help='path of cache directory', default=None)
    parser.add_argument('--use_gpu', action='store_true', help='use gpu')
    parser.add_argument("--include", type=str, nargs="+", default=["mean"], choices=["mean", "per_tok", "bos", "contacts"], help="which representations to return")
    parser.add_argument("--repr_layers", type=int, nargs="+", default=[-3, -2, -1], help="which layers to extract features from, default is [-3, -2, -1] which means the last three layers")
    args = parser.parse_args()
    plm = PlmEmbed(
        args.fasta_file,
        args.workdir,
        model_name=args.model_name,
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        use_gpu=args.use_gpu,
        include=args.include,
        repr_layers=args.repr_layers,
    )
    plm.extract(plm.fasta_file, 
    repr_layers=plm.repr_layers, 
    model_path=plm.model_path, 
    model_name=plm.model_name, 
    use_gpu=plm.use_gpu, 
    include=plm.include, 
    output_dir=plm.cache_dir,
    )

