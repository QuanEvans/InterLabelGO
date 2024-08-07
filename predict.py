import os, argparse
import multiprocessing as mp
import torch
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle, shutil, json

from InterLabelGO_pred import InterLabelGO_pipline
from alignment_knn import AlignmentKNN
from utils import obo_tools
from settings import settings_dict as settings
oboTools = obo_tools.ObOTools(
    go_obo=settings['obo_file'],
    obo_pkl=settings['obo_pkl_file']
)

def get_current_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def record_time(func):
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        start_time = time.time()
        print("start %s: %s" % (func_name, get_current_time()))
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time = round(end_time - start_time, 2)
        print("end %s: %s" % (func_name, get_current_time()))
        print("Total time for %s: %s seconds" % (func_name, total_time))
        print('----------------------------------------------------------------\n')
        return result
    return wrapper

InterLabel_weight_dict = {'BPO': 0.94, 'CCO': 0.55, 'MFO': 0.76} 
a_weight_dict = {'BPO': 0.5, 'CCO': 0.4, 'MFO': 0.25}
k_weight_dict = {'BPO': 2, 'CCO': 5, 'MFO': 4}

# ZLPR_PTF1_GOF1
a_weight_dict = {'BPO': 0.45, 'CCO': 0.45, 'MFO': 0.25}
k_weight_dict = {'BPO': 1.5, 'CCO': 4, 'MFO': 5}
# old zlpr_ptf1_gof1
# a_weight_dict = {'BPO': 0.55, 'CCO': 0.45, 'MFO': 0.25}
# k_weight_dict = {'BPO': 5, 'CCO': 4, 'MFO': 5}

# ZLPR_GOF1
a_weight_dict = {'BPO': 0.5, 'CCO': 0.5, 'MFO': 0.2}
k_weight_dict = {'BPO': 3.5, 'CCO': 5, 'MFO': 5}
# old zlpr_gof1
# a_weight_dict = {'BPO': 0.75, 'CCO': 0.55, 'MFO': 0.2}
# k_weight_dict = {'BPO': 4, 'CCO': 5, 'MFO': 5}


class MainPipline:

    def __init__(self,
        working_dir:str,
        fasta_file:str,
        num_threads:int=mp.cpu_count(),
        device:str='cuda',
        top_terms:int=500, # number of top terms to be keeped in the prediction
        aspects:list=['BPO', 'CCO', 'MFO'], # aspects of model
        pred_batch_size:int=512,
        InterLabel_min_weight:float=0.1,
        no_align:bool=False,
        no_dnn:bool=False,
        cache_dir:str=None,
        seqid_combine:bool=False,
        embed_batch_size:int=4096, # note this might take around 15GB of vram, if you don't have enough vram, you can set this to 2048
        # DO NOT MODIFY BELOW THIS LINE if you want to use the pretrained model
        continue_from_last:bool=False,
        alignment_database_dir:str=settings['alignment_db'],
        alignment_labels_dir:str=settings['alignment_labels'],
        model_dir:str=settings['MODEL_CHECKPOINT_DIR'],
    ):
        self.working_dir = os.path.abspath(working_dir)
        self.fasta_file = os.path.abspath(fasta_file)
        self.num_threads = num_threads
        self.device = device
        self.top_terms = top_terms
        self.aspects = aspects
        self.pred_batch_size = pred_batch_size
        self.embed_batch_size = embed_batch_size
        self.no_align = no_align
        self.no_dnn = no_dnn
        self.cache_dir = cache_dir
        self.combine_tsv = os.path.join(self.working_dir, 'InterLabelGO+.tsv')

        self.alignment_database_dir = alignment_database_dir
        self.alignment_labels_dir = alignment_labels_dir
        self.InterLabel_min_weight = InterLabel_min_weight
        self.model_dir = os.path.abspath(model_dir)
        self.seqid_combine = seqid_combine

        self.InterLabelGO_workdir = os.path.join(self.working_dir, 'InterLabelGO_pred')
        self.AlignmentKNN_workdir = os.path.join(self.working_dir, 'AlignmentKNN_pred')
        self.continue_from_last = continue_from_last
        if not self.continue_from_last:
            shutil.rmtree(self.InterLabelGO_workdir, ignore_errors=True)
            shutil.rmtree(self.AlignmentKNN_workdir, ignore_errors=True)
        if not os.path.exists(self.InterLabelGO_workdir):
            os.makedirs(self.InterLabelGO_workdir)
        if not os.path.exists(self.AlignmentKNN_workdir):
            os.makedirs(self.AlignmentKNN_workdir)

    @record_time
    def InterLabelGO_pred(self):
        InterLabelGO_pipline(
            working_dir=self.InterLabelGO_workdir,
            fasta_file=self.fasta_file,
            pred_batch_size=self.pred_batch_size,
            device=self.device,
            top_terms=self.top_terms,
            embed_batch_size=self.embed_batch_size,
            model_dir=self.model_dir,
            aspects=self.aspects,
            cache_dir=self.cache_dir,
        ).main()
    
    @record_time
    def AlgignmentKNN_pred(self):
        AlignmentKNN(
            working_dir=self.AlignmentKNN_workdir,
            fasta_file=self.fasta_file,
            Database_dir=self.alignment_database_dir,
            Database_label=self.alignment_labels_dir,
            num_threads=self.num_threads,
            aspects=self.aspects,
            continue_from_last=self.continue_from_last,
        ).main()
    
    @record_time
    def combine_results(self, InterLabelGO_tsv, AlignmentKNN_tsv, max_seqid_tsv):
        dnn = pd.read_csv(InterLabelGO_tsv, sep='\t')

        align = pd.read_csv(AlignmentKNN_tsv, sep='\t')
        seqid = pd.read_csv(max_seqid_tsv, sep='\t')

        with open(self.combine_tsv, 'w') as f:
            colum_names = ['EntryID', 'term', 'score', 'max_seqid', 'aspect', 'go_term_name']
            f.write('\t'.join(colum_names) + '\n')

        for aspect in self.aspects:
            # select current aspect
            aspect_dnn = dnn[dnn['aspect'] == aspect]
            aspect_align = align[align['aspect'] == aspect]
            aspect_seqid = seqid[seqid['aspect'] == aspect]
            # drop aspect columns
            aspect_dnn = aspect_dnn.drop(columns=['aspect'])
            aspect_align = aspect_align.drop(columns=['aspect'])
            aspect_seqid = aspect_seqid.drop(columns=['aspect'])
            # convert to dict
            seqid_dict = dict(zip(aspect_seqid['EntryID'], aspect_seqid['max_seqid']))

            # only keep EntryID, term, score
            aspect_dnn = aspect_dnn[['EntryID', 'term', 'score']]
            aspect_align = aspect_align[['EntryID', 'term', 'score']]

            # merge dnn, align
            merged_df = pd.merge(aspect_dnn, aspect_align,on=['EntryID', 'term'], how='outer', suffixes=('_dnn', '_align'))

            # merged_df['score_dnn'].fillna(0, inplace=True)
            # merged_df['score_align'].fillna(0, inplace=True)
            merged_df = merged_df.fillna({'score_dnn': 0, 'score_align': 0})

            # extract all scores
            entry_ids = merged_df['EntryID'].values
            score_dnn = merged_df['score_dnn'].values
            score_align = merged_df['score_align'].values

            args = zip(entry_ids, score_dnn, score_align)
            scores =[]
            for arg in args:
                scores.append(self.combine_score(arg,aspect=aspect, seqid_dict=seqid_dict, seqid_combine=self.seqid_combine))
            merged_df['score'] = scores
            merged_df = self.parent_propagation(merged_df)
            merged_df['aspect'] = aspect
            
            # only keep EntryID, aspect, term, score
            merged_df = merged_df[merged_df['score'] >= 0.01]
            merged_df['max_seqid'] = merged_df['EntryID'].map(seqid_dict)
            merged_df = merged_df[['EntryID', 'term', 'score', 'max_seqid', 'aspect']]
            # round the score to 3 decimal places
            merged_df['score'] = merged_df['score'].apply(lambda x: round(x, 3))
            merged_df['max_seqid'] = merged_df['max_seqid'].apply(lambda x: round(x, 3))
            # sort by EntryID, aspect, score
            merged_df = merged_df.sort_values(['EntryID', 'aspect','score'], ascending=[True, True, False])
            # only keep the top 500 terms
            merged_df = merged_df.groupby(['EntryID', 'aspect']).head(self.top_terms)
            merged_df['go_term_name'] = merged_df['term'].apply(lambda x: oboTools.goID2name(x))
            merged_df.to_csv(self.combine_tsv, sep='\t', index=False, mode='a', header=False)

    # def parent_propagation(self, df: pd.DataFrame):
    #     '''
    #     propagate the prediction to the parent terms
    #     df.columns = ['EntryID', 'term', 'score']
    #     '''
    #     # Convert to dict, where key is the EntryID, value dict of term and score
    #     #df_dict = df.groupby('EntryID').apply(lambda x: x.set_index('term')['score'].to_dict()).to_dict()
    #     df_dict = df.groupby('EntryID', group_keys=False)[df.columns].apply(lambda x: x.set_index('term')['score'].to_dict()).to_dict()
        
    #     # Propagate the prediction to the parent terms
    #     result_dict = {}
    #     for EntryID, term_score in df_dict.items():
    #         result_dict[EntryID] = oboTools.backprop_cscore(term_score, min_cscore=0.001)
        
    #     # Convert back to dataframe
    #     rows = []
    #     for EntryID, terms_scores in result_dict.items():
    #         for term, score in terms_scores.items():
    #             rows.append({'EntryID': EntryID, 'term': term, 'score': score})
        
    #     result_df = pd.DataFrame(rows)
    #     return result_df

    def parent_propagation(self, df: pd.DataFrame):
        '''
        Propagate the prediction to the parent terms
        df.columns = ['EntryID', 'term', 'score']
        '''
        # Convert to a pivot table
        pivot_df = df.pivot(index='EntryID', columns='term', values='score')
        
        # Apply backprop_cscore to each row
        result = pivot_df.apply(lambda row: pd.Series(oboTools.backprop_cscore(row.to_dict(), min_cscore=0.001)), axis=1)
        
        # Reset the index and melt the DataFrame back to long format
        result_df = result.reset_index().melt(id_vars='EntryID', var_name='term', value_name='score')
        
        # Remove rows where score is NaN (terms not present for that EntryID)
        result_df = result_df.dropna(subset=['score'])
        
        return result_df

    def combine_score(self, args, aspect:str=None, seqid_dict:dict=None, seqid_combine:bool=False):
        entry_id, score_dnn, score_align = args

        if not seqid_combine:
            score = score_dnn * InterLabel_weight_dict[aspect] + score_align * (1 - InterLabel_weight_dict[aspect])
            return score
    
        dnn_weight = 0.5
        seqid = seqid_dict.get(entry_id, 0.01)
        #score = score_dnn * (dnn_weight/ (dnn_weight + seqid)) + score_align * (seqid / (dnn_weight + seqid))
        # a = 0.33
        # k = 3
        a = a_weight_dict[aspect]
        k = k_weight_dict[aspect]
        w = a + (1 - a) * np.exp(-k * seqid)
        score = score_dnn * w + score_align * (1 - w)
        return score
 
    def main(self):
        if not self.no_dnn:
            #self.InterLabelGO_pred()
            pass
        if not self.no_align:
            #self.AlgignmentKNN_pred()
            pass
        if not self.no_dnn and not self.no_align:
            AlignmentKNN_tsv=os.path.join(self.AlignmentKNN_workdir, 'AlignmentKNN.tsv')
            max_seqid_tsv = os.path.join(self.AlignmentKNN_workdir, 'seqid.tsv')
            InterLabelGO_tsv=os.path.join(self.InterLabelGO_workdir, 'InterLabelGO.tsv')
            print('combining results...')
            self.combine_results(InterLabelGO_tsv, AlignmentKNN_tsv, max_seqid_tsv)
            # cp interlabel results into
            #shutil.copy(InterLabelGO_tsv, os.path.join(self.working_dir, 'InterLabelGO.tsv'))


if __name__ == "__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument('-w', '--working_dir', type=str, help='working directory', required=True)
    parser.add_argument('-f', '--fasta_file', type=str, help='fasta file', required=True)
    parser.add_argument('-t', '--num_threads', type=int, help='number of threads', default=mp.cpu_count())
    parser.add_argument('-top', '--top_terms', type=int, help='number of top terms to be keeped in the prediction', default=500)
    parser.add_argument('-m','--model_dir', type=str, default=settings['MODEL_CHECKPOINT_DIR'], help='directory to saved models')
    parser.add_argument("-c", "--continue_from_last", help="continue from the last unfinished file", action="store_true")
    parser.add_argument('--use_gpu', action='store_true', help='use gpu')
    parser.add_argument('--no_align', action='store_true', help='do not perform alignment')
    parser.add_argument('--no_dnn', action='store_true', help='do not perform dnn prediction')
    parser.add_argument('--aspect', type=str, nargs='+', default=['BPO', 'CCO', 'MFO'], choices=['BPO', 'CCO', 'MFO'], help='aspects of model to predict')
    parser.add_argument('--cache', type=str, default=None, help='cache dir that store precomputed embeddings')
    parser.add_argument('--seqid', action='store_true', help='use seqid to combine results')
    args = parser.parse_args()
    working_dir = os.path.abspath(args.working_dir)
    fasta_file = os.path.abspath(args.fasta_file)
    cache = os.path.abspath(args.cache) if args.cache is not None else None

    if args.use_gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    MainPipline(
        working_dir=working_dir,
        fasta_file=fasta_file,
        num_threads=args.num_threads,
        device=device,
        top_terms=args.top_terms,
        model_dir=args.model_dir,
        no_align=args.no_align,
        no_dnn=args.no_dnn,
        aspects=args.aspect,
        cache_dir=cache,
        seqid_combine=args.seqid,
        continue_from_last=args.continue_from_last
    ).main()

