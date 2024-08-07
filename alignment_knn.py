import os, sys, subprocess
from Bio import SeqIO
import multiprocessing as mp
from multiprocessing import Pool
import shutil
import pandas as pd
from tqdm import tqdm
import argparse
import numpy as np
import json

# the following package is from local
from utils import obo_tools, run_diamond
from settings import settings_dict as settings

docstring = """
Blast/Diamond alginment based KNN Protein Gene Ontology Annotation Pipeline (AlignmentKNN)

example usage:
    python alignment_knn.py -w /home/username/workdir -f /home/username/seq.fasta -d /home/username/database -g /home/username/goa -t 8

keywords arguments:
    -w, --workdir: working directory for the pipline
    -f, --fasta: path of fasta file
    -d, --database: dir path of database, should include three sub directories: BPO, MFO, CCO, each sub directory should contain a fasta file named AlignmentKNN.fasta
    -g, --goa: dir path of goa labels file, should include three files: BPO_Term, MFO_Term, CCO_Term, each file should have multiple lines, each line contains a protein name and its go terms separated by tab, the go terms separated by comma
    -t, --threads: number of threads, default is the number of cores of the machine
"""


class AlignmentKNN:
    ### class variables ###
    global oboTools
    oboTools = obo_tools.ObOTools(
        go_obo=settings['obo_file'],
        obo_pkl=settings['obo_pkl_file']
    )
    global blast_hits
    blast_hits = dict()
    global aspect_goTerm_labels
    aspect_goTerm_labels = dict()
    ### END ###

    def __init__(self,
        working_dir:str,
        fasta_file:str, 
        Database_dir:str, 
        Database_label:str, 
        num_threads:int=mp.cpu_count(), 
        aspects=['BPO', 'CCO', 'MFO'],
        continue_from_last:bool=False,
    ):
        
        self.working_dir = os.path.abspath(working_dir)
        self.result_file = os.path.join(self.working_dir, "AlignmentKNN.tsv")
        self.seqid_file = os.path.join(self.working_dir, "seqid.tsv")
        self.fasta_file = os.path.abspath(fasta_file)
        self.aspects = aspects # biological process, molecular function, cellular component
        self.tmp = os.path.join(self.working_dir, "tmp")
        self.DatabaseDir = os.path.abspath(Database_dir)
        self.GOA = os.path.abspath(Database_label)
        self.num_threads = num_threads
        self.continue_from_last = continue_from_last
        if not continue_from_last:
            shutil.rmtree(self.working_dir, ignore_errors=True)
            os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.tmp, exist_ok=True)
    
    def get_seq_dict(self, fast_file:str)->dict:
        """read fasta file and return a dict

        Args:
            fast_file (str): path of fasta file

        Returns:
            dict: dict of fasta file, key is protein name, value is protein sequence
        """
        seq_dict = {}
        for record in SeqIO.parse(fast_file, "fasta"):
            seq_dict[record.id] = str(record.seq)
        return seq_dict

    def read_labels(self, filename:str)->dict:
        """read labels from file
            file should have multiple lines, each line contains a protein name and its go terms separated by tab
            the go terms separated by comma
            example:
                protein1 go1,go2,go3

        Args:
            filename (str): path to the label file

        Returns:
            dict: dict of labels, key is protein name, value is a set of go terms
        """
        with open(filename, 'r') as f:
            go_term_dict = dict()
            for line in f:
                line = line.strip().split()
                if len(line) > 1:
                    name = line[0]
                    go_terms = line[1:]
                    go_terms = [i.split(',') for i in go_terms]
                    go_terms = [j for i in go_terms for j in i]
                    go_term_dict[name] = set(go_terms)
                else:
                    print(line)
        return go_term_dict
    
    def create_database(self, terms_tsv:str, fasta_file:str):
        """create database for alignment
        terms_tsv should be a tsv file that contains EntryID, aspect, term separated by tab

        Args:
            terms_tsv (str): path of terms_tsv
            fasta_file (str): path of fasta file
        """
        terms_df = pd.read_csv(terms_tsv, sep='\t')
        seq_dict = self.get_seq_dict(fasta_file)

        if not os.path.exists(self.DatabaseDir):
            os.makedirs(self.DatabaseDir, exist_ok=True)
        if not os.path.exists(self.GOA):
            os.makedirs(self.GOA, exist_ok=True)

        for aspect in self.aspects:
            if not os.path.exists(os.path.join(self.DatabaseDir, aspect)):
                os.makedirs(os.path.join(self.DatabaseDir, aspect), exist_ok=True)

            cur_aspect_df = terms_df[terms_df['aspect'] == aspect].copy()
            # group by EntryID, apply set to term
            cur_aspect_df = cur_aspect_df.groupby(['EntryID'])['term'].apply(set).reset_index()
            # backprop parent
            cur_aspect_df['term'] = cur_aspect_df['term'].apply(lambda x: oboTools.backprop_set(x))
            cur_aspect_df['term'] = cur_aspect_df['term'].apply(lambda x: ','.join(x))
            unique_entryids = set(cur_aspect_df['EntryID'].unique())
            if len(unique_entryids) == 0:
                raise Exception(f"Error: no {aspect} terms found in {terms_tsv}, please make sure aspect is one of BPO, MFO, CCO")

            # write to file
            labels_path = os.path.join(self.GOA, f"{aspect}_Term")
            cur_aspect_df.to_csv(labels_path, sep='\t', index=False, header=False)

            aspect_fasta_path = os.path.join(self.DatabaseDir, aspect, 'AlignmentKNN.fasta')
            with open(aspect_fasta_path, 'w') as f:
                for entryid in unique_entryids:
                    if entryid in seq_dict:
                        f.write(f">{entryid}\n{seq_dict[entryid]}\n")
                    else:
                        print(f"Warning: {entryid} not found in {fasta_file}, skip it", file=sys.stderr)
            
            # create and diamond_db
            diamond_db_cmd = f'{settings["diamond_path"]} makedb --in {aspect_fasta_path} -d {os.path.join(self.DatabaseDir, aspect, "AlignmentKNN")}'
            print('creating diamond_db...')
            subprocess.run(diamond_db_cmd, shell=True, cwd=self.DatabaseDir, check=True)

    
    def annotate_protein(self, name:str):
        """
        annotate a protein

        the cscore is calculated as follows:
            cscore = sum of bitscore of all hits that have the term / sum of bitscore of all hits

        Args:
            sub_dir (str): protein directory, is a sub directory of SAGP directory
            name (str): protein name
            sequence (str): protein sequence
        
        Returns:
            name (str): protein name, return name for better multiprocessing
            protein_result (dict): key is aspect, value is a dict, key is go term, value is cscore
            protein_aspect_max_seqid_dict (dict): key is aspect, value is the max_ident of all blast hits
        """
        # annotate protein

        protein_result = dict()

        protein_aspect_max_seqid_dict = dict()

        # for each aspect, get blast hits
        for aspect in self.aspects:
            # get the database labels, key is protein name, value is a set of go terms
            cur_apsect_protein_label = aspect_goTerm_labels[aspect]
            # get current protein blast hits
            cur_blast_hits = blast_hits[aspect].get(name, None)
            if cur_blast_hits is None:
                # no hits for current protein so skip
                continue
            # weighted the bitscore by identity
            cur_blast_hits['ident'] = cur_blast_hits.apply(lambda x: x['nident'] / max(x['qlen'], x['slen']), axis=1)

            # find the max_ident of all blast hits and store it in protein_aspect_max_seqid_dict
            # length weighted
            top_ident = cur_blast_hits.sort_values(by='bitscore', ascending=False).head(5)
            s3 = top_ident.apply(lambda x: x['nident'] / min(x['qlen'], x['slen']), axis=1)
            sum_bitscore = sum(top_ident['bitscore'])
            mean_top_ident = sum(s3 * top_ident['bitscore']) / sum_bitscore

            protein_aspect_max_seqid_dict[aspect] = mean_top_ident
            
            #cur_blast_hits['ident'] = cur_blast_hits.apply(lambda x: x['nident'] /  x['qlen'], axis=1)
            cur_blast_hits['bitscore'] = cur_blast_hits['bitscore'] * cur_blast_hits['ident']
            cur_blast_hits = cur_blast_hits[['target','bitscore']]
            # conver to dict
            target_bitscore_dict = dict(zip(cur_blast_hits['target'], cur_blast_hits['bitscore']))

            # annotate protein

            # get all go terms
            term_list = set()
            for target in target_bitscore_dict:
                if target in cur_apsect_protein_label:
                    term_list.update(cur_apsect_protein_label[target])

            result_dict = dict()
            for term in term_list:
                sum_bitscore = sum(target_bitscore_dict.values()) # sum of all bitscore from all targets
                term_sum_bitscore = sum(target_bitscore_dict[target] for target in target_bitscore_dict if term in cur_apsect_protein_label[target]) # sum of all bitscore from all targets that have current term
                result_dict[term] = term_sum_bitscore / sum_bitscore # cscore
            
            # update protein_result
            result_dict = oboTools.backprop_cscore(result_dict, min_cscore = 0.001)
            protein_result[aspect] = result_dict
        
        return name, protein_result, protein_aspect_max_seqid_dict
                  
    def get_blast_hits(self, workdir:str, fasta_file:str, threads:int=mp.cpu_count())->None:
        """run alignment, use fasta file as query, AlignmentKNN as database, extract all hits

        Args:
            workdir (str): working directory for saving results
            fasta_file (str): path of fasta file
            threads (int, optional): number of core to use. Defaults to mp.cpu_count().
        """

        for aspect in self.aspects:
            final_fasta_file = fasta_file
            hits_tsv_name = os.path.join(workdir, f"{aspect}_blast.tsv")

            database = os.path.join(self.DatabaseDir, aspect, 'AlignmentKNN.dmnd')
            if not os.path.exists(database):
                raise Exception(f"Error: {database} does not exist in {self.DatabaseDir}")
            
            processed_file = None

            if os.path.exists(hits_tsv_name):
                # check whether it is complete
                first_line = open(hits_tsv_name).readline()
                if first_line.startswith("query"):
                    continue
                
                columns = ["query", "target", "bitscore", "pident" , "evalue", "qlen", "slen", "nident"]
                processed = pd.read_csv(hits_tsv_name, sep='\t', names=columns)
                # check if tmp tsv file exists
                tmp_tsv_name = os.path.join(self.tmp, f"{aspect}_tmp.tsv")
                if os.path.exists(tmp_tsv_name):
                    processed_tmp = pd.read_csv(tmp_tsv_name, sep='\t', names=columns)
                    # if exists, then merge
                    processed = pd.concat([processed_tmp, processed])
            
                processed_id = processed['query'].unique().tolist()
                fasta_dict = self.get_seq_dict(fasta_file)
                # remove the last num core query, because it is we don't know whether it is complete
                processed_id = processed_id[:-mp.cpu_count()]
                processed_id_set = set(processed_id)
                fasta_dict = {k: v for k, v in fasta_dict.items() if k not in processed_id_set}
                with open(os.path.join(self.tmp, f"{aspect}_tmp.fasta"), "w") as f:
                    for name, seq in fasta_dict.items():
                        f.write(f">{name}\n{seq}\n")
                final_fasta_file = os.path.join(self.tmp, f"{aspect}_tmp.fasta")

                processed = processed[processed['query'].isin(processed_id_set)]
                processed.to_csv(tmp_tsv_name, sep='\t', index=False, header=False)
                processed_file = tmp_tsv_name

            run_diamond.run_diamond(final_fasta_file, database, output_file=hits_tsv_name, threads=threads, processed_file=processed_file)
    
    def read_hits(self, hits_tsv_name):
        blast_hits_aspect_df_grouped_dict = {}
        current_query = None
        current_group = []       
        with open(hits_tsv_name, 'r') as f:
            header = f.readline().strip().split('\t')
            rest_lines = f.readlines()
            for line in tqdm(rest_lines, total=len(rest_lines), ascii=' >='):
                row = line.strip().split('\t')
                row[2:] = [float(i) for i in row[2:]]
                # Extract 'query' value from the row
                # Assuming 'query' is the first column; adjust the index if needed
                query = row[0]
                
                if current_query is None:
                    current_query = query
                    
                if query == current_query:
                    current_group.append(row)
                else:
                    if current_query in blast_hits_aspect_df_grouped_dict:
                        print('Warning: duplicated query found in blast hits, this hits file is not sorted', file=sys.stderr)
                        blast_hits_aspect_df_grouped_dict[current_query] = \
                            pd.concat([blast_hits_aspect_df_grouped_dict[current_query], pd.DataFrame(current_group, columns=header)])
                    else:
                        blast_hits_aspect_df_grouped_dict[current_query] = pd.DataFrame(current_group, columns=header)
                        
                    current_query = query
                    current_group = [row]
                
        # at the end of the file, add the last group
        if current_query in blast_hits_aspect_df_grouped_dict:
            print('Warning: duplicated query found in blast hits, this hits file is not sorted', file=sys.stderr)
            blast_hits_aspect_df_grouped_dict[current_query] = \
                pd.concat([blast_hits_aspect_df_grouped_dict[current_query], pd.DataFrame(current_group, columns=header)])
        else:
            blast_hits_aspect_df_grouped_dict[current_query] = pd.DataFrame(current_group, columns=header)

        return blast_hits_aspect_df_grouped_dict

    def main(self):
        """
        Pipline for AlignmentKNN
        """

        # read fasta file
        seq_dict = self.get_seq_dict(self.fasta_file)

        # run blast
        self.get_blast_hits(self.working_dir, self.fasta_file, threads=self.num_threads)

        # read blast hits
        # blast_hits is already a global variable
        for aspect in self.aspects:
            print(f"reading {aspect} blast hits...")
            hits_tsv_name = os.path.join(self.working_dir, f"{aspect}_blast.tsv")
            blast_hits[aspect] = self.read_hits(hits_tsv_name)

        # read labels
        # aspect_goTerm_labels is already a global variable
        for aspect in self.aspects:
            label_path = os.path.join(self.GOA, f"{aspect}_Term")
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Error: {aspect} label ({label_path}) does not exist in {self.GOA}")
            aspect_goTerm_labels[aspect] = self.read_labels(label_path)

        # annotate proteins using multiprocessing
        args_list = []
        for name in seq_dict:
            args_list.append(name)

        print("Generating Alginment KNN results...")
        with Pool(self.num_threads) as p:
            final_result = p.map(self.annotate_protein, tqdm(args_list, total=len(args_list), ascii=' >='))
        
        # # write results
        with open(self.result_file, 'w') as f, open(self.seqid_file, 'w') as f2:
            f.write('EntryID\tterm\tscore\tmax_seqid\taspect\tgo_term_name\n')
            f2.write('EntryID\tmax_seqid\taspect\n')
            for name, protein_result, protein_aspect_max_seqid_dict in final_result:
                for aspect in protein_result.keys():
                    max_seqid = protein_aspect_max_seqid_dict[aspect]
                    f2.write(f"{name}\t{max_seqid}\t{aspect}\n")
                    for term, score in protein_result[aspect].items():
                        f.write(f"{name}\t{term}\t{score}\t{max_seqid}\t{aspect}\t{oboTools.goID2name(term)}\n")
        
        # remove tmp directory
        shutil.rmtree(self.tmp, ignore_errors=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=docstring, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-w", "--workdir", help="working directory", required=True)
    parser.add_argument("-f", "--fasta", help="path of fasta file", required=True)
    parser.add_argument("-d", "--database", help="dir path of database", default=os.path.join(file_dir, settings['alignment_db']))
    parser.add_argument("-g", "--goa", help="dir path of goa labels file", default=os.path.join(file_dir, settings['alignment_labels']))
    parser.add_argument("-t", "--threads", help="number of threads", default=mp.cpu_count(), type=int)
    parser.add_argument("-c", "--continue_from_last", help="continue from the last unfinished file", action="store_true")
    parser.add_argument('--aspects', type=str, nargs='+', default=['BPO', 'CCO', 'MFO'], choices=['BPO', 'CCO', 'MFO'], help='aspects of GO terms to be predicted')
    args = parser.parse_args()
    workdir = os.path.abspath(args.workdir)
    fasta_file = os.path.abspath(args.fasta)
    database = os.path.abspath(args.database)
    goa = os.path.abspath(args.goa)
    num_threads = args.threads
    alignmentKNN = AlignmentKNN(workdir, fasta_file, database, goa, num_threads, aspects=args.aspects, continue_from_last=args.continue_from_last)
    alignmentKNN.main()