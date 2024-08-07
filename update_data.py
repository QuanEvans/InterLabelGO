import os, shutil
import json
import pandas as pd
from utils.obo_tools import ObOTools
from Bio import SeqIO

file_dir = os.path.dirname(os.path.realpath(__file__))
SETTINGS_FILE = os.path.join(file_dir, 'settings.json')
settings = json.load(open(SETTINGS_FILE, 'r'))


def main():
    """
    Download the newest UNIPROT GOA file and preprocess to the csv and fasta
    """
    tmp_dir = os.path.join(file_dir, settings['tmp_dir'])
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    # Download the obo file
    print('Downloading the newest obo file')
    obo_file_path = os.path.join(file_dir, settings['obo_file'])
    obo_pkl_path = os.path.join(file_dir, settings['obo_pkl_file'])
    # remove the old obo file if exists
    if os.path.exists(obo_file_path):
        os.remove(obo_file_path)
    if os.path.exists(obo_pkl_path):
        os.remove(obo_pkl_path)
    os.system(f"wget http://purl.obolibrary.org/obo/go.obo -O {obo_file_path}")
    oboTools = ObOTools()
    oboTools.init_obo(go_obo=obo_file_path)
    aspect_map_dict = {
        "P": "BPO",
        "F": "MFO",
        "C": "CCO",
    }
    # Download the newest UNIPROT GOA file
    print('Downloading the newest UNIPROT GOA file')
    os.system(f"wget ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/goa_uniprot_all.gaf.gz -O {tmp_dir}/goa.gaf.gz")
    # only grep exp data
    print('Extracting the exp data')
    os.system(f'zcat {tmp_dir}/goa.gaf.gz | grep "^UniProtKB" | grep -P "(\tEXP\t)|(\tIDA\t)|(\tIPI\t)|(\tIMP\t)|(\tIGI\t)|(\tIEP\t)|(\tTAS\t)|(\tIC\t)|(\tHTP\t)|(\tHDA\t)|(\tHMP\t)|(\tHGI\t)|(\tHEP\t)" > {tmp_dir}train.gaf')

    # download sequence
    print('Downloading the sequence')
    os.system(f'wget ftp://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.fasta.gz -O {tmp_dir}/uniprot_sprot.fasta.gz')
    os.system(f'zcat {tmp_dir}/uniprot_sprot.fasta.gz >> {tmp_dir}/uniprot_sprot.fasta')

    # filter the train.gaf
    df = pd.read_csv(f'{tmp_dir}/train.gaf', sep='\t', header=None)
    # exclude the not
    df = df[~df[6].str.startswith('NOT')]
    df = df[df[0] == "UniProtKB"]
    df = df[[1,4,8]]
    col_names = ['EntryID', 'term', 'aspect']
    df.columns = col_names
    df['aspect'] = df['aspect'].apply(lambda x: aspect_map_dict[x])
    all_entry_ids = set(df['EntryID'].unique())
    present_ids = set()

    # get fasta sequence from sprot fasta
    print('Extracting the fasta sequence')
    train_seqs = f'{tmp_dir}/train_seq.fasta'
    with open(train_seqs, 'w') as f:
        for record in SeqIO.parse(f'{tmp_dir}/uniprot_sprot.fasta', 'fasta'):
            if '|' in record.id:
                record.id = record.id.split('|')[1]
            if record.id in all_entry_ids:
                f.write(f'>{record.id}\n')
                f.write(f'{record.seq}\n')
                present_ids.add(record.id)
    
    # missing ids
    missing_ids = all_entry_ids - present_ids
    # download the missing ids
    print(f'Downloading {len(missing_ids)} missing ids')
    for target in missing_ids:
        # curl "https://rest.uniprot.org/unisave/$target?format=fasta&versions=1" |cut -f1,2 -d'|' |sed 's/>sp|/>/g' |sed 's/>tr|/>/g' 
        os.system(f'curl "https://rest.uniprot.org/unisave/{target}?format=fasta&versions=1" |cut -f1,2 -d\'|\' |sed \'s/>sp|/>/g\' |sed \'s/>tr|/>/g\' >> {train_seqs}')
    
    # get all id from the train_seqs
    all_ids = set()
    for record in SeqIO.parse(train_seqs, 'fasta'):
        all_ids.add(record.id)
    
    df = df[df['EntryID'].isin(all_ids)]
    bp_df = df[df['aspect'] == 'BPO']
    cc_df = df[df['aspect'] == 'CCO']
    mf_df = df[df['aspect'] == 'MFO']
    # only keep the term that is in the obo file
    bp_df = bp_df[bp_df['term'].apply(lambda x: x in oboTools.term2aspect)]
    cc_df = cc_df[cc_df['term'].apply(lambda x: x in oboTools.term2aspect)]
    mf_df = mf_df[mf_df['term'].apply(lambda x: x in oboTools.term2aspect)]
    # collapse
    bp_df = bp_df.groupby(['EntryID', 'aspect'])['term'].apply(set).reset_index()
    cc_df = cc_df.groupby(['EntryID', 'aspect'])['term'].apply(set).reset_index()
    mf_df = mf_df.groupby(['EntryID', 'aspect'])['term'].apply(set).reset_index()
    # apply the oboTools
    bp_df['term'] = bp_df['term'].apply(lambda x: oboTools.backprop_terms(x))
    bp_df = bp_df.explode('term')
    cc_df['term'] = cc_df['term'].apply(lambda x: oboTools.backprop_terms(x))
    cc_df = cc_df.explode('term')
    mf_df['term'] = mf_df['term'].apply(lambda x: oboTools.backprop_terms(x))
    mf_df = mf_df.explode('term')
    prop_df = pd.concat([bp_df, mf_df, cc_df])
    prop_df.to_csv(f'{tmp_dir}/train_terms.tsv', index=False, sep='\t')
    
    # cp the train_sequences and train_terms to raw data
    parent_dir = os.path.dirname(os.path.join(file_dir, settings['train_terms_tsv']))
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    shutil.copy(f"{tmp_dir}/train_seq.fasta", os.path.join(file_dir, settings['train_seqs_fasta']))
    shutil.copy(f"{tmp_dir}/train_terms.tsv", os.path.join(file_dir, settings['train_terms_tsv']))
    # rm the tmp_dir
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

if __name__ == '__main__':
    main()