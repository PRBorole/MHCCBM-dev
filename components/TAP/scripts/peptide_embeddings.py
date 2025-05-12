# load packages
import numpy as np
import pandas as pd
import argparse
import sys
import os
import pickle

import torch
import esm
from tqdm import tqdm

# Get the path of the Python script
current_dir = os.path.abspath(os.path.dirname(__file__))
# Exclude script name at the end
current_dir = os.path.split(current_dir)[0]+'/'
sys.path.append(current_dir+'/../../')

from src.utils import *


parser = argparse.ArgumentParser(description="usage help",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
TAP_df = pd.read_csv(current_dir+'/../../data/TAP/classification_DS868.csv',sep='\t')
peptides = list(TAP_df['peptide'].unique())

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

# Prepare data 
data = [(peptide, peptide) for i,peptide in enumerate(peptides)]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

embedding_dict = {} 
for idx, batch_token in tqdm(enumerate(batch_tokens)):
    embedding_dict[batch_labels[idx]] = get_esm_embedding(model, batch_tokens[idx:idx+1], mean=True)
    
    
with open(current_dir+'/../../data/TAP/classification_peptides_esm1b.pkl','wb') as f:
    pickle.dump(embedding_dict, f)
    
print(len(embedding_dict))

