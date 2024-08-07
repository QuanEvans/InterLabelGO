import random as rd
import numpy as np
import torch
import os, json
import shutil
from torch.utils.data import DataLoader
import pickle, argparse
import scipy.sparse as ssp

from Network.model import InterlabelGODataset, InterLabelResNet
from Network.model_utils import InterLabelLoss, EarlyStop, FmaxMetric, Trainer
from utils.obo_tools import ObOTools
from plm import PlmEmbed
from settings import settings_dict as settings
from settings import training_config, add_res_dict

# training_config = {
#     'activation':'gelu',
#     'layer_list':[2048],
#     'embed_dim':2560,
#     'dropout':0.3,
#     'epochs':200,
#     'batch_size':512,
#     'pred_batch_size':8124*4,
#     'learning_rate':0.001,
#     'num_models': 5,
#     'patience':10,
#     'min_epochs':20,
#     'seed':12,
#     'repr_layers': [34, 35, 36],
#     'log_interval':1,
#     'eval_interval':1,
#     'monitor': 'both',
# }

# # seed_dict = {
# #     0: 80399,
# #     1: 61392,
# #     2: 13533,
# #     3: 86992,
# #     4: 70825,
# # }

# add_res_dict = {
#     'BPO':True,
#     'CCO':False,
#     'MFO':False,
# } # whether to add residual connections or not

oboTools = obo_tools.ObOTools(
    go_obo=settings['obo_file'],
    obo_pkl=settings['obo_pkl_file']
)

def seed_everything(seed):
    rd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_term_list(file_name):
    term_list = []
    with open(file_name) as f:
        for line in f:
            term_list.append(line.rstrip())
    return term_list

def read_ia(filename):
    ia_dict = dict()
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split()
            if len(line)!= 2:
                raise ValueError('IA file format error')
            ia_dict[line[0]] = line[1]
    return ia_dict

def get_vec2go(term_list):
    vec2go_dict = dict()
    for i, go_term in enumerate(term_list):
        vec2go_dict[i] = go_term
    return vec2go_dict

def calculate_weight_matrix(term_list:list, vec2go_dict:dict, ia_dict:dict):
    weigth_array = np.zeros(len(term_list))
    for i in range(len(term_list)):
        go_term = vec2go_dict[i]
        ia_score = ia_dict.get(go_term, 0)
        weigth_array[i] = float(ia_score) + 1e-16
    return weigth_array

def main(  
        train_data_dir:str=settings['TRAIN_DATA_CLEAN_DIR'],
        embed_feature_dir:str=settings['embedding_dir'],
        model_dir:str=settings['MODEL_CHECKPOINT_DIR'],
        ia_file:str=settings['ia_file'],
        device:str='cuda', 
        aspects:list=['BPO', 'CCO', 'MFO'],
        training_config:dict=training_config,
        add_res_dict:dict=add_res_dict,
    ):
    seed = training_config['seed']
    seed_everything(training_config['seed'])
    ia_dict = read_ia(ia_file) # read the IA.txt file, load the ia_dict
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for aspect in aspects:

        # training_config['learning_rate'] = 0.001
        # if aspect == "BPO":
        #     training_config['learning_rate'] = 0.001

        aspect_model_dir = os.path.join(model_dir, aspect)
        if not os.path.exists(aspect_model_dir):
            os.makedirs(aspect_model_dir)
        
        term_list = read_term_list(os.path.join(train_data_dir, aspect, 'term_list.txt')) # read the term_list.txt file, load the go_term_list
        vec2go = get_vec2go(term_list) # get the vec2go_dict
        weight_array = calculate_weight_matrix(term_list, vec2go, ia_dict) # calculate the weight_array
        weight_tensor = torch.from_numpy(weight_array).to(device) # convert the weight_array to weight_tensor
        child_array = oboTools.generate_child_matrix(term_list) # generate the child_array
        # save child_array
        child_array_path = os.path.join(aspect_model_dir, 'child_matrix_ssp.npz')
        # convert it to ssp format
        child_array_ssp = ssp.csr_matrix(child_array)
        ssp.save_npz(child_array_path, child_array_ssp)

        child_tensor = torch.from_numpy(child_array).to(device) # convert the child_array to child_tensor

        for i in range(training_config['num_models']):
            # seed = seed_dict[i]
            # seed_everything(seed)

            save_name = os.path.join(aspect_model_dir, f'model_{i}.pt')
            model = InterLabelResNet(
                aspect=aspect,
                layer_list=training_config['layer_list'],
                embed_dim=training_config['embed_dim'],
                dropout=training_config['dropout'],
                activation=training_config['activation'],
                go_term_list=term_list,
                add_res=add_res_dict[aspect],
                seed=seed,
            )
            ## create Dataloader
            train_dataset = InterlabelGODataset(
                features_dir=embed_feature_dir,
                names_npy=os.path.join(train_data_dir, aspect, f'{aspect}_train_names_fold{i}.npy'),
                labels_npy=os.path.join(train_data_dir, aspect, f'{aspect}_train_labels_fold{i}.npz'),
                repr_layers=training_config['repr_layers'],
            )
            val_dataset = InterlabelGODataset(
                features_dir=embed_feature_dir,
                names_npy=os.path.join(train_data_dir, aspect, f'{aspect}_valid_names_fold{i}.npy'),
                labels_npy=os.path.join(train_data_dir, aspect, f'{aspect}_valid_labels_fold{i}.npz'),
                #names_npy=os.path.join(train_data_dir, aspect, f'{aspect}_test_names.npy'),
                #labels_npy=os.path.join(train_data_dir, aspect, f'{aspect}_test_labels.npz'),
                repr_layers=training_config['repr_layers'],
            )
            train_loader = DataLoader(train_dataset, batch_size = training_config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size = training_config['pred_batch_size'], shuffle=False)

            loss_fn = InterLabelLoss(device=device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=training_config['learning_rate'], weight_decay=1e-3, amsgrad=True, betas=(0.9, 0.999), eps=1e-6)
            metric = FmaxMetric(weight_matrix=weight_tensor, child_matrix=child_tensor, device=device)

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                learning_rate=training_config['learning_rate'],
                loss_fn=loss_fn,
                optimizer=optimizer,
                metric=metric,
                device=device,
                epochs=training_config['epochs'],
                patience=training_config['patience'],
                min_epochs=training_config['min_epochs'],
                early_stopping=True,
                aspect=aspect,
                weight_matrix=weight_tensor,
                child_matrix=child_tensor,
                log_interval=training_config['log_interval'],
                eval_interval=training_config['eval_interval'],
                monitor=training_config['monitor'],
            )
            print(f'Training model {i} for aspect {aspect}')
            save_model, best_loss, best_f1, num_epoch = trainer.fit() # save_model is a boolean value indicating whether to save the model or not
            if save_model:
                model.save_config(save_name)
                log_file = os.path.join(aspect_model_dir, f'log_{i}.pkl')
                log_dict = dict()
                log_dict['best_loss'] = best_loss
                log_dict['best_f1'] = best_f1
                log_dict['num_epoch'] = num_epoch
                log_dict['training_history'] = trainer.history
                log_dict['training_config'] = training_config
                with open(log_file, 'wb') as f:
                    pickle.dump(log_dict, f)
            else:
                print(f'Not saving model')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default=settings['TRAIN_DATA_CLEAN_DIR'], help='directory to save / saved network training data')
    parser.add_argument('--embed_feature', type=str, default=settings['embedding_dir'], help='directory to save / saved embed features')
    parser.add_argument('--model_dir', type=str, default=settings['MODEL_CHECKPOINT_DIR'], help='directory to save models')
    parser.add_argument('--ia_file', type=str, default=settings['ia_file'], help='file containing the information content of GO terms')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--aspects', type=str, nargs='+', default=['BPO', 'CCO', 'MFO'], choices=['BPO', 'CCO', 'MFO'], help='aspects of model to train')
    
    args = parser.parse_args()
    args.train_data = os.path.abspath(args.train_data)
    args.embed_feature = os.path.abspath(args.embed_feature)
    args.model_dir = os.path.abspath(args.model_dir)
    
    if not torch.cuda.is_available():
        args.device = 'cpu'
    
    main(
        train_data_dir=args.train_data,
        embed_feature_dir=args.embed_feature,
        model_dir=args.model_dir,
        ia_file=args.ia_file,
        device=args.device,
        aspects=args.aspects,
    )




    



