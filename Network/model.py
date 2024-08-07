import os, random, pickle
import numpy as np
import torch
import scipy.sparse as ssp
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from typing import List, Union
import multiprocessing as mp
from tqdm import tqdm

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class InterlabelGODataset(Dataset):
    def __init__(self,
        features_dir:str,
        names_npy:str,
        labels_npy:str=None,
        repr_layers:list=[34, 35, 36],
        low_memory:bool=False,
    ):

        self.features_dir = features_dir
        self.names_npy = names_npy
        self.repr_layers = repr_layers
        self.low_memory = low_memory
        self.feature_cache = dict()

        if labels_npy is None:
            self.prediction = True
        else:
            self.prediction = False
            self.labels_npy = labels_npy

        # load names, labels
        self.names = np.load(self.names_npy)
        if not self.prediction:
            self.labels = self.load_labels(self.labels_npy)
        if not self.low_memory:
            for name in tqdm(self.names):
                self.feature_cache[name] = self.load_feature(name)

    
    def load_labels(self, labels_npy:str)->np.ndarray:
        """
        Load labels from npy or npz file.

        Args:
            labels_npy (str): path to npy or npz file

        Raises:
            Exception: Unknown label file format

        Returns:
            np.ndarray: labels
        """
        if labels_npy.endswith(".npy"):
            labels = np.load(labels_npy)
        elif labels_npy.endswith(".npz"):
            labels = ssp.load_npz(labels_npy).toarray()
        else:
            raise Exception("Unknown label file format")
        labels = torch.from_numpy(labels).float()
        return labels

    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        name = self.names[idx]
        if name not in self.feature_cache and not self.low_memory:
            feature = self.load_feature(name)
            self.feature_cache[name] = feature
        else:
            feature = self.feature_cache[name]
            
        if self.prediction:
            return name, feature
        else:
            label = self.labels[idx]
            return name, feature, label
    
    def load_feature(self, name:str)->np.ndarray:
        """
        Load feature from npy file.

        Args:
            name (str): name of the feature

        Returns:
            np.ndarray: feature
        """
        features = np.load(os.path.join(self.features_dir, name + '.npy'),allow_pickle=True).item()
        final_features = []
        for layer in self.repr_layers:
            final_features.append(features['mean'][layer])
        final_features = np.concatenate(final_features)
        return torch.from_numpy(final_features).float()


class InterlabelGODatasetWindow(Dataset):
    def __init__(self,
        features_dir:str,
        fasta_dict:dict,
        repr_layers:list=[34, 35, 36],
        window_size:int=50,
    ):
        self.features_dir = features_dir
        self.fasta_dict = fasta_dict
        self.repr_layers = repr_layers
        self.window_size = window_size
        self.data_list = self.load_data()

    def load_data(self):
        data_dict = {}
        for name, seq in self.fasta_dict.items():
            # truncate the sequence to the first 1000 amino acids, because the esm model only accept 1000 amino acids
            if len(seq) > 1000:
                seq = seq[:1000]
            features = np.load(os.path.join(self.features_dir, name + '.npy'),allow_pickle=True).item()
            seq_len = len(seq)
            # create windows
            for i in range(0, seq_len-self.window_size+1):
                start = i
                end = i + self.window_size
                final_features = []
                for layer in self.repr_layers:
                    final_features.append(features['per_tok'][layer][start:end].mean(axis=0))
                final_features = np.concatenate(final_features)
                final_features = torch.from_numpy(final_features).float()
                data_dict[name + '_' + str(start) + '-' + str(end)] = final_features
        return list(data_dict.items())

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        name, feature = self.data_list[idx]
        return name, feature

class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class InterLabelResNet(nn.Module):
    def __init__(self, 
        aspect:str=None, # aspect of the GO terms
        layer_list:list=[1024], # layers of dnn network, example [512, 256, 128]
        embed_dim:int=2560, # dim of the embedding protein language model
        go_term_list:List[str]=[], # list of GO terms for prediction
        dropout:float=0.3, # dropout rate
        activation:str='elu', # activation function
        seed:int=42, # random seed
        prediction_mode:bool=False, # if True, the model will output the prediction of the GO terms
        add_res:bool=False,
        ):
        super(InterLabelResNet, self).__init__()
        self.aspect = aspect
        self.layer_list = layer_list
        self.embed_dim = embed_dim
        self.go_term_list = go_term_list
        self.vec2go_dict = self.get_vec2go()
        self.class_num = len(go_term_list)
        self.dropout = dropout
        self.activation = activation
        self.seed = seed
        self.prediction_mode = prediction_mode
        self.add_res = add_res

        # bach normalization for the input
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        self.bn3 = nn.BatchNorm1d(embed_dim)

        # Define DNN branches
        self.branch1 = self._build_dnn_branch(embed_dim)
        self.branch2 = self._build_dnn_branch(embed_dim)
        self.branch3 = self._build_dnn_branch(embed_dim)
        
        # concat dense layer
        self.concat_layer = nn.Sequential(
            nn.Linear(layer_list[-1]*3, (layer_list[-1])),
            self.get_activation(activation),
            nn.Dropout(dropout),
            nn.BatchNorm1d((layer_list[-1])),
        )


        if self.add_res:
            self.res = Residual(
                nn.Sequential(
                    nn.Linear(layer_list[-1], layer_list[-1]),
                    self.get_activation(activation),
                    nn.Dropout(0.1),
                    nn.BatchNorm1d((layer_list[-1])),
                )
            )

        self.output_layer = nn.Sequential(
            nn.Linear((layer_list[-1]), self.class_num),
            #nn.Sigmoid(),
        )

        # initialize weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def _build_dnn_branch(self, embed_dim):
        layers = []
        for i, layer in enumerate(self.layer_list):
            layers.append(nn.Linear(embed_dim if i == 0 else self.layer_list[i - 1], layer))
            layers.append(self.get_activation(self.activation))
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.BatchNorm1d(layer))
        return nn.Sequential(*layers) # * is used to unpack the list for the nn.Sequential

    def forward(self, inputs):
        x1 = inputs[:, :self.embed_dim]
        x2 = inputs[:, self.embed_dim:2*self.embed_dim]
        x3 = inputs[:, 2*self.embed_dim:]


        # batch normalization for each branch
        x1 = self.bn1(x1)
        x2 = self.bn2(x2)
        x3 = self.bn3(x3)

        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        x3 = self.branch3(x3)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.concat_layer(x)

        if self.add_res:
            x = self.res(x)

        y_pred = self.output_layer(x)

        if self.prediction_mode:
            y_pred = torch.sigmoid(y_pred)

        return y_pred
    
    def get_vec2go(self):
        vec2go_dict = dict()
        for i, go_term in enumerate(self.go_term_list):
            vec2go_dict[i] = go_term
        return vec2go_dict
    
    def get_activation(self, activation:str):
        activation = activation.lower()
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leakyrelu':
            return nn.LeakyReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'gelu':
            return nn.GELU()
        else:
            raise ValueError('activation function not supported')
    
    def save_config(self, save_path:str):
        config = {
            'aspect': self.aspect, 
            'layer_list': self.layer_list, 
            'embed_dim': self.embed_dim, 
            'go_term_list': self.go_term_list, 
            'dropout': self.dropout, 
            'activation': self.activation,
            'seed': self.seed, 
            'add_res': self.add_res,
            'state_dict': self.state_dict(),
        }
        torch.save(config, save_path)
        

    @staticmethod
    def load_config(save_path:str):
        config = torch.load(save_path, map_location=torch.device('cpu'))
        model = InterLabelResNet(
            aspect=config['aspect'], 
            layer_list=config['layer_list'], 
            embed_dim=config['embed_dim'], 
            go_term_list=config['go_term_list'], 
            dropout=config['dropout'], 
            activation=config['activation'],
            seed=config['seed'], 
            add_res=config['add_res'],
        )
        # load the state_dict, but only match the keys
        model.load_state_dict(config['state_dict'], strict=False)
        return model

