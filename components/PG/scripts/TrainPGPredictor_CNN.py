import sys
from tqdm import tqdm
import os
import json
import time
import argparse

import pandas as pd
import numpy as np

from Bio import SeqIO
import esm

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from torchinfo import summary

from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler


# Get the path of the Python script
current_dir = os.path.abspath(os.path.dirname(__file__))
# Exclude script name at the end
current_dir = os.path.split(current_dir)[0]+'/'
sys.path.append(current_dir+'/../../src/')

from utils import *
from utils_torch import * 
from MHCCBM import *
from PGPredictor_CNN import *

parser = argparse.ArgumentParser(description="usage help",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--config_path", help="path to config file")
parser.add_argument("--save_model", help="boolean, save model")
parser.add_argument("--fold", help="fold idx for 5 CV")
parser.add_argument("--flank", help="flanking length")

args = parser.parse_args()
config_path = vars(args)

config_path = args.config_path
save_model = int(args.save_model)
fold = int(args.fold)
flank = int(args.flank)

seed = 42
set_seed(seed)
print("Seed: ", seed)


# load json file
with open(config_path) as jsonfile:
    config_dict = json.load(jsonfile)
print("config: ", config_dict)

config = config_dict['config']

embedding_df = pd.read_csv(current_dir+'/../../data/PG/esm1b/flank'+str(flank)+'_peptides_esm1b.csv',index_col=0)

# Make  and y
X = embedding_df.drop(['peptide','hit'],axis=1).to_numpy()
y = embedding_df['hit'].to_numpy()

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X.squeeze())
X = torch.tensor(X, dtype=torch.float32).reshape(X.shape[0],1,X.shape[1])

#### model training
input_size = X.shape[-1] #embedding size for esm2_t33_650M_UR50D (allele + peptide)

# Calculate class weights
labels_tensor = torch.tensor(y, dtype=torch.int16)
class_counts = torch.bincount(labels_tensor)
pos_weight = class_counts[0]/class_counts[1]
pos_weight = pos_weight.to(dtype=torch.float32)
print(pos_weight)


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
(train, test) = [(train, test) for (train, test) in kf.split(X, y)][fold]

# Create dataset and dataloaders
train_dataset = ProteinSequenceDataset(X[train], y[train])
test_dataset = ProteinSequenceDataset(X[test], y[test])

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)


#### model init
input_size = X.shape[2] #embedding size for esm2_t33_650M_UR50D (peptide)
model = PGPredictor(input_size, config['hidden_channels'], 
                             config['kernel_size'], config['pool_kernel_size'], config['dropout_p'])


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
result['fold'] = [fold] 

#### performance
f1, auroc, auprc = model.eval_dataset(train_loader)
result['f1_train'], result['auroc_train'], result['auprc_train'] = [f1], [auroc], [auprc]

f1, auroc, auprc = model.eval_dataset(test_loader)
result['f1_test'], result['auroc_test'], result['auprc_test'] = [f1], [auroc], [auprc]

result_df = pd.DataFrame(result)
print(result)
print(result_df)

#### save model and results
summary(model, input_size=(config['batch_size'],1,input_size))

if save_model:
    torch.save(model, current_dir+'/results/flank'+str(flank)+'_PGpred.pt')
    
   
