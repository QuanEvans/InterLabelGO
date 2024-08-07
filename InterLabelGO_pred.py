import os, argparse
import torch
from Bio import SeqIO
import numpy as np
import pickle
import scipy.sparse as ssp
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from torch.utils.data import DataLoader
import math, json

from Network.model import InterlabelGODataset, InterLabelResNet
from Network.model_utils import Predictor
from utils.obo_tools import ObOTools
from plm import PlmEmbed
from settings import settings_dict as settings

# the following package is from local
from utils import obo_tools
oboTools = obo_tools.ObOTools(
    go_obo=settings['obo_file'],
    obo_pkl=settings['obo_pkl_file']
)

class InterLabelGO_pipline:
    def __init__(self,
        working_dir:str,
        fasta_file:str,
        pred_batch_size:int=512,
        device:str='cuda',
        top_terms:int=500, # number of top terms to be keeped in the prediction
        aspects:list=['BPO', 'CCO', 'MFO'], # aspects to predict
        cache_dir:str=None,
        ## the following parameters should be fixed if you want to use the pretrained model
        repr_layers:list=[34, 35, 36],
        embed_batch_size:int=4096, # note this might take around 15GB of vram, if you don't have enough vram, you can set this to 2048
        embed_model_name:str="esm2_t36_3B_UR50D",
        embed_model_path:str=settings['esm3b_path'],
        include:list=['mean'],
        model_dir:str=settings['MODEL_CHECKPOINT_DIR'],
    ) -> None:
        self.working_dir = os.path.abspath(working_dir)
        self.fasta_file = os.path.abspath(fasta_file)
        self.pred_batch_size = pred_batch_size
        
        if not torch.cuda.is_available():
            device = 'cpu'
        self.device = device
        self.top_terms = top_terms
        self.aspects = aspects
        self.cache_dir = cache_dir

        self.repr_layers = repr_layers
        self.embed_model_name = embed_model_name
        self.embed_model_path = embed_model_path
        self.include = include
        self.embed_batch_size = embed_batch_size

        self.model_dir = os.path.abspath(model_dir)
        self.result_file = os.path.join(self.working_dir, 'InterLabelGO.tsv')


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
    
    def get_embed_features(self):
        Embed = PlmEmbed(
            fasta_file=self.fasta_file,
            working_dir=self.working_dir,
            repr_layers=self.repr_layers,
            model_name=self.embed_model_name,
            model_path=self.embed_model_path,
            use_gpu=('cuda' in self.device),
            include=self.include,
            cache_dir=self.cache_dir,
        )
        print("Extracting embeding features")
        Embed.extract(
            fasta_file=self.fasta_file,
            model_name=self.embed_model_name,
            model_path=self.embed_model_path,
            use_gpu=('cuda' in self.device),
            repr_layers=self.repr_layers,
            include=self.include,
            batch_size=self.embed_batch_size,
            model_type='esm',
            )
        feature_dir = Embed.cache_dir
        return feature_dir
    
    def create_name_npy(self):
        fasta_dict = self.parse_fasta(self.fasta_file)
        name_npy_path = os.path.join(self.working_dir, 'names.npy')
        names = np.array(list(fasta_dict.keys()))
        np.save(name_npy_path, names)
        return name_npy_path

    def predict(self, feature_dir:str):
        name_npy_path = self.create_name_npy() # create working_dir/names.npy file for DataLoader
        predictor = Predictor(
            model=None,
            PredictLoader=None,
            device=self.device,
            child_matrix=None,
            back_prop=True,
        )
        PredictDataset = InterlabelGODataset(
        features_dir=feature_dir,
        names_npy=name_npy_path,
        repr_layers=self.repr_layers,
        labels_npy=None,# set to because we are doing inference
        )
        PredictLoader = DataLoader(PredictDataset, batch_size=self.pred_batch_size, shuffle=False, num_workers=0)
        predictor.update_loader(PredictLoader)

        result_file = self.result_file
        columns = ['EntryID','term','score','aspect', 'go_term_name']
        with open(result_file, 'w') as f:
            f.write('\t'.join(columns))
            f.write('\n')
        seeds_dict = dict()
        for aspect in self.aspects:
            aspect_model_dir = os.path.join(self.model_dir,aspect)
            if not os.path.exists(aspect_model_dir):
                print(f'No model found for {aspect}')
                continue
            models = os.listdir(aspect_model_dir)
            models = [os.path.join(aspect_model_dir, model) for model in models if model.endswith('.pt')]

            if len(models) == 0:
                print(f"No model found in {aspect_model_dir}")
                continue
            child_matrix = ssp.load_npz(os.path.join(aspect_model_dir, 'child_matrix_ssp.npz')).toarray()
            predictions = []
            names = None
            print(f"Predicting {aspect}")

            # # only use one model for now
            # models = models[:1]

            for model_path in tqdm(models, desc=f'generate {aspect} prediction', ascii=' >='):
                model = InterLabelResNet.load_config(model_path)
                seed = model.seed
                if aspect not in seeds_dict:
                    seeds_dict[aspect] = {}
                seeds_dict[aspect][model_path] = seed
                # # save model again
                # model.save_config(model_path)
                model = model.to(self.device)
                predictor.update_model(model,child_matrix)
                predictor.back_prop = False
                protein_ids, y_preds = predictor.predict()
                predictions.append(y_preds)
                if names is None:
                    names = protein_ids

            predictions = sum(predictions)/len(predictions)

            term_list = model.go_term_list

            # convert to dataframe
            df = pd.DataFrame(predictions, columns=term_list, index=names)
            df = df.stack().reset_index()
            df.columns = ['EntryID', 'term', 'score']
            #df = self.parent_propagation(df)
            df = df[df['score'] > 0.001]
            # sort by name then score
            df = df.sort_values(by=['EntryID','score'], ascending=[True, False])
            df = df.sort_values(by=['EntryID','score'], ascending=[True, False])
            df['aspect'] = aspect

            # only keep the top 500 terms
            df = df.groupby(['EntryID', 'aspect']).head(self.top_terms)
            df = df[['EntryID', 'term', 'score', 'aspect']]
            df['go_term_name'] = df['term'].apply(lambda x: oboTools.goID2name(x))
            # write to tsv file
            df.to_csv(result_file, index=False, sep='\t', mode='a', header=False)
            #print(seeds_dict)

    def parent_propagation(self, df: pd.DataFrame):
        '''
        propagate the prediction to the parent terms
        df.columns = ['EntryID', 'term', 'score']
        '''
        # Convert to dict, where key is the EntryID, value dict of term and score
        df_dict = df.groupby('EntryID').apply(lambda x: x.set_index('term')['score'].to_dict()).to_dict()
        
        # Propagate the prediction to the parent terms
        result_dict = {}
        for EntryID, term_score in tqdm(df_dict.items(), desc='propagate prediction', ascii=' >='):
            result_dict[EntryID] = oboTools.backprop_cscore(term_score, min_cscore=0.001)
        
        # Convert back to dataframe
        rows = []
        for EntryID, terms_scores in result_dict.items():
            for term, score in terms_scores.items():
                rows.append({'EntryID': EntryID, 'term': term, 'score': score})
        
        result_df = pd.DataFrame(rows)
        return result_df
        

    def main(self):
        feature_dir = self.get_embed_features()
        self.predict(feature_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # example usage: python InterLabelGO_pred.py -w example -f example/example.fasta --use_gpu
    parser.add_argument('-w', '--working_dir', type=str, help='working directory', required=True)
    parser.add_argument('-f', '--fasta_file', type=str, help='fasta file', required=True)
    parser.add_argument('-top', '--top_terms', type=int, help='number of top terms to be keeped in the prediction', default=500)
    parser.add_argument('-m', '--model_dir', type=str, help='model directory', default=settings['MODEL_CHECKPOINT_DIR'])
    parser.add_argument('--esm_path', type=str, help='esm model path', default=settings['esm3b_path'])
    parser.add_argument('--use_gpu', action='store_true', help='use gpu')
    parser.add_argument('--aspect', type=str, nargs='+', default=['BPO', 'CCO', 'MFO'], choices=['BPO', 'CCO', 'MFO'], help='aspects of model to predict')
    parser.add_argument('--cache_dir', type=str, help='cache directory', default=None)
    args = parser.parse_args()
    working_dir = os.path.abspath(args.working_dir)
    fasta_file = os.path.abspath(args.fasta_file)
    model_dir = os.path.abspath(args.model_dir)
    esm_path = os.path.abspath(args.esm_path)
    device = 'cuda' if args.use_gpu else 'cpu'
    InterLabelGO_pipline(
        working_dir=working_dir,
        fasta_file=fasta_file,
        device=device,
        top_terms=args.top_terms,
        aspects=args.aspect,
        model_dir=model_dir,
        embed_model_path=esm_path,
        cache_dir=args.cache_dir,
    ).main()