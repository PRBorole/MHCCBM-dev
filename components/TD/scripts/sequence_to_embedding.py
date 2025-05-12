# load packages
import numpy as np
import pandas as pd
import pickle

import torch
import esm

# load full TD dataframe
TD_full_df = pd.read_csv('/exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/TD/Data/processed_data/TD_full.csv',index_col=0)
TD_full_df

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

# Prepare data 
data = [(v['HLA_full'], v['Sequence']) for k,v in TD_full_df[['HLA_full','Sequence']].T.to_dict().items()]
batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

# Extract per-residue representations (on CPU)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
token_representations = results["representations"][33]

# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
sequence_representations = []
for i, tokens_len in enumerate(batch_lens):
    sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

with open('/exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/TD/Data/processed_data/hla_esm1b.pkl','wb') as f:
    pickle.dump(sequence_representations,f)