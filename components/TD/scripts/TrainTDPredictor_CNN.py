# load packages
import numpy as np
import pandas as pd
import seaborn as sns
import time
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import torch
import os
import esm
from tqdm import tqdm
import pickle
import json

from torchinfo import summary

from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split, LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import sys
import argparse

# Get the path of the Python script
current_dir = os.path.abspath(os.path.dirname(__file__))
# Exclude script name at the end
current_dir = os.path.split(current_dir)[0]+'/'
sys.path.append(current_dir+'/../../src/')

from utils import *
from utils_torch import * 
from MHCCBM import *
from TDPredictor_CNN import *


parser = argparse.ArgumentParser(description="usage help",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--config_path", help="path to config file")
parser.add_argument("--save_model", help="boolean, save model")

args = parser.parse_args()
config_path = vars(args)

config_path = args.config_path
save_model = int(args.save_model)


# load json file
with open(config_path) as jsonfile:
    config_dict = json.load(jsonfile)
config = config_dict['config']
print("config: ", config_dict)

seed = config['seed']
set_seed(seed)
print("Seed: ", seed)

# load full TD dataframe
TD_full_df = pd.read_csv(current_dir+'./../../data/TD/processed_data//TD_full.csv',index_col=0)
TD_full_df = TD_full_df.rename(columns={'HLA_full':'allele'})

# load embeddings
with open(current_dir+'./../../data/TD/processed_data//allele_esm1b.pkl','rb') as f:
    embedding_dict = pickle.load(f)
    
## Combined embeddings and y
merged_df = pd.DataFrame({'allele':embedding_dict.keys(),
                          'embedding':embedding_dict.values()}).merge(TD_full_df, 
                                                                      on='allele')[['embedding','MFI_ratio']]

X = torch.cat(merged_df['embedding'].to_list())
y = merged_df['MFI_ratio'].to_numpy()

# Scale the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X.squeeze())
X = torch.tensor(X, dtype=torch.float32)

# Scale y
y = np.where(y>2.0, 1, 0)

# Split the data
train_sequences, test_sequences, train_labels, test_labels = train_test_split(X, y, test_size=0.1, 
                                                                              random_state=seed,stratify=y)


kf = StratifiedKFold(n_splits=10, shuffle=True)
result_df = []


for idx, (train, test) in enumerate(kf.split(train_sequences, train_labels)):
    # Create dataset and dataloaders
    train_dataset = ProteinSequenceDataset(train_sequences.reshape(train_sequences.shape[0],1,-1)[train], train_labels[train])
    test_dataset = ProteinSequenceDataset(train_sequences.reshape(train_sequences.shape[0],1,-1)[test], train_labels[test])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Calculate class weights
    labels_tensor = torch.tensor(train_labels[train])
    class_counts = torch.bincount(labels_tensor)
    pos_weight = class_counts[0]/class_counts[1]
    print(pos_weight)

    #### model init
    input_size = train_sequences.shape[1] #embedding size for esm2_t33_650M_UR50D (peptide)
    model = TDPredictor(input_size, 
                     hidden_channels=config['hidden_channels'],
                     dropout_p=config['dropout_p'],
                     kernel_size = config['kernel_size'],
                     pool_kernel_size = config['pool_kernel_size'])


    # model training
    start = time.time()
    epoch = model.train_loop(train_loader=train_loader, 
                             valid_loader=test_loader, 
                             test_loader=None,
                             pos_weight=pos_weight,
                             config_dict=config_dict)
    end = time.time()

    time_elapsed = end-start
    print("Time taken: ", time_elapsed)

    result = {'time_elapsed': [time_elapsed]}
    result['epoch'] = [epoch]
    result['fold'] = [idx] 

    #### performance
    f1, auroc, auprc = model.eval_dataset(train_loader)
    result['f1_train'], result['auroc_train'], result['auprc_train'] = [f1], [auroc], [auprc]


    f1, auroc, auprc = model.eval_dataset(test_loader)
    result['f1_test'], result['auroc_test'], result['auprc_test'] = [f1], [auroc], [auprc]

    print(result)
    result_df = result_df + [pd.DataFrame(result)]

    #### save model and results
    summary(model, input_size=(config['batch_size'],1,input_size))
    
result_df = pd.concat(result_df)
pd.DataFrame(result_df).to_csv(current_dir+'/results/hyperparameter_tuning_result/CNN/'+config_dict['name']+'.csv')

