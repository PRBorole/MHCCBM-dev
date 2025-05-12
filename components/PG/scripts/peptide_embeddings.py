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
PG_df = pd.read_csv(current_dir+'/../../data/PG/PG.csv')
peptides_ls = list(PG_df['peptide'].unique())

model, alphabet, model_state = esm.pretrained._load_model_and_alphabet_core_v2(torch.load(current_dir+'/../../../../esm_models/esm2_t33_650M_UR50D.pt'))
batch_converter = alphabet.get_batch_converter()

model.eval()  # disables dropout for deterministic results

indice_ls = [i for i in range(0, 300100, 100)]

for range_sec in tqdm(range(1, len(indice_ls))):
    peptides = peptides_ls[indice_ls[range_sec-1]: indice_ls[range_sec]]

    # Prepare data 
    data = [(peptide, peptide) for i,peptide in enumerate(peptides)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    embeddings = get_esm_embedding(model, batch_tokens, mean=True)
    embedding_dict = {batch_labels[idx]:t for idx, t in enumerate(embeddings)}


    with open(current_dir+'/../../data/PG/esm1b/flank0/flank0_peptides_esm1b_'+str(range_sec)+'.pkl','wb') as f:
        pickle.dump(embedding_dict, f)

    print(len(embedding_dict))