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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import wandb

# Get the path of the Python script
current_dir = os.path.abspath(os.path.dirname(__file__))
# Exclude script name at the end
current_dir = os.path.split(current_dir)[0]+'/'
sys.path.append(current_dir+'/../../src/')

from utils import *
from utils_torch import * 
from MHCCBM import *
from TAPPredictor import *

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
print("config: ", config_dict)

config = config_dict['config']

# load X and y
with open(current_dir+'/../../data/TAP/X.pkl','rb') as f:
    X = pickle.load(f)
f.close()

with open(current_dir+'/../../data/TAP/y.pkl','rb') as f:
    y = pickle.load(f)
f.close()

# Scale the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X.squeeze())
X = torch.tensor(X, dtype=torch.float32)

# Split the data
train_sequences, temp_sequences, train_labels, temp_labels = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
valid_sequences, test_sequences, valid_labels, test_labels = train_test_split(temp_sequences, temp_labels, test_size=0.5, random_state=seed, stratify=temp_labels)

# Create dataset and dataloaders
train_dataset = ProteinSequenceDataset(train_sequences, train_labels)
valid_dataset = ProteinSequenceDataset(valid_sequences, valid_labels)
test_dataset = ProteinSequenceDataset(test_sequences, test_labels)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)


#### model training
input_size = X.shape[1] #embedding size for esm2_t33_650M_UR50D (allele + peptide)
model = TAPPredictor(input_size, config['hidden_size'])

# Calculate class weights
labels_tensor = torch.tensor(train_labels)
class_counts = torch.bincount(labels_tensor)
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum()  # Normalize

# Convert to a tensor
class_weights = class_weights.to(dtype=torch.float32)
pos_weight = 1/class_weights[1]

start = time.time()
model.train_loop(train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader, config_dict=config_dict, pos_weight=pos_weight)
end = time.time()

time_elapsed = end-start
print("Time taken: ", time_elapsed)

result = {'time_elapsed': [time_elapsed]}
result['random_seed'] = [seed] 

#### performance
f1, auroc, auprc = model.eval_dataset(train_loader)
result['f1_train'], result['auroc_train'], result['auprc_train'] = [f1], [auroc], [auprc]

f1, auroc, auprc = model.eval_dataset(valid_loader)
result['f1_valid'], result['auroc_valid'], result['auprc_valid'] = [f1], [auroc], [auprc]

f1, auroc, auprc = model.eval_dataset(test_loader)
result['f1_test'], result['auroc_test'], result['auprc_test'] = [f1], [auroc], [auprc]

print(result)

#### save model and results
summary(model, input_size=(config['batch_size'],1,input_size))

pd.DataFrame(result).to_csv(current_dir+'/results/hyperparameter_tuning_result/'+config_dict['name']+'.csv')

if save_model:  
    torch.save(model.state_dict(), current_dir+'/results/hyperparameter_tuning_model/'+config_dict['name']+'.pt')
    
   