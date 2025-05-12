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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import wandb


# Get the path of the Python script
current_dir = os.path.abspath(os.path.dirname(__file__))
# Exclude script name at the end
current_dir = os.path.split(current_dir)[0]+'/'
sys.path.append(current_dir+'/../../src/')

from utils import *
from utils_torch import * 
from MHCCBM import *
from TAPPredictor_CNN_2outputs import *


parser = argparse.ArgumentParser(description="usage help",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--config_path", help="path to config file")
parser.add_argument("--save_model", help="boolean, save model")

args = parser.parse_args()
config_path = vars(args)

config_path = args.config_path
save_model = int(args.save_model)

seed = 42
set_seed(42)
print("Seed: ", seed)

# load json file
with open(config_path) as jsonfile:
    config_dict = json.load(jsonfile)
config = config_dict['config']
print("config: ", config_dict)

train_df = pd.read_csv(current_dir+'/../../data/TAP/DeepTAP_train_test_split/train.csv')
test_df = pd.read_csv(current_dir+'/../../data/TAP/DeepTAP_train_test_split/test.csv')

with open(current_dir+'/../../data/TAP/classification_peptides_esm1b.pkl', 'rb') as f:
    peptide_embedding_dict = pickle.load(f)
f.close()

train_sequences = torch.cat([peptide_embedding_dict[p] for p in train_df['peptide'].to_list()])
train_labels = train_df['label'].to_numpy()
test_sequences = torch.cat([peptide_embedding_dict[p] for p in test_df['peptide'].to_list()])
test_labels = test_df['label'].to_numpy()

# Scale the data
scaler = StandardScaler()
train_sequences = scaler.fit_transform(train_sequences.squeeze())
train_sequences = torch.tensor(train_sequences, dtype=torch.float32)
train_sequences = train_sequences.reshape((train_sequences.shape[0],1,-1))

test_sequences = scaler.fit_transform(test_sequences.squeeze())
test_sequences = torch.tensor(test_sequences, dtype=torch.float32)
test_sequences = test_sequences.reshape((test_sequences.shape[0],1,-1))

# Create dataset and dataloaders
train_dataset = ProteinSequenceDataset(train_sequences, train_labels)
test_dataset = ProteinSequenceDataset(test_sequences, test_labels)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)


# Calculate class weights
labels_tensor = torch.tensor(train_labels)
class_counts = torch.bincount(labels_tensor)
pos_weight = class_counts[0]/class_counts

print(pos_weight)

#### model init
input_size = train_sequences.shape[2] #embedding size for esm2_t33_650M_UR50D (peptide)
model = TAPPredictor(input_size, 
                     hidden_channels=config['hidden_channels'], 
                     dropout_p=config['dropout_p'])


# model training
start = time.time()
model.train_loop(train_loader=train_loader, valid_loader=None, pos_weight=pos_weight,
                 test_loader=test_loader, config_dict=config_dict)
end = time.time()

time_elapsed = end-start
print("Time taken: ", time_elapsed)

result = {'time_elapsed': [time_elapsed]}
result['random_seed'] = [seed] 

#### performance
f1, auroc, auprc = model.eval_dataset(train_loader)
result['f1_train'], result['auroc_train'], result['auprc_train'] = [f1], [auroc], [auprc]


f1, auroc, auprc = model.eval_dataset(test_loader)
result['f1_test'], result['auroc_test'], result['auprc_test'] = [f1], [auroc], [auprc]

print(result)


#### save model and results
summary(model, input_size=(config['batch_size'],1,input_size))

pd.DataFrame(result).to_csv(current_dir+'/results/hyperparameter_tuning_result/'+config_dict['name']+'.csv')

if save_model:  
    torch.save(model.state_dict(), current_dir+'/results/hyperparameter_tuning_model/'+config_dict['name']+'.pt')
    
   