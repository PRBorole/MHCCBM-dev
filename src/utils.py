# load packages
import numpy as np
import pandas as pd
import pickle
from Bio import SeqIO
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
import esm

def set_seed(seed):
    """Set seed for reproducibility.

    Args:
    seed (int): Seed value.
    """
    # Set seed for random module
    random.seed(seed)
    
    # Set seed for numpy
    np.random.seed(seed)
    
    # If you're using PyTorch, you can set seed for torch as well
    # Uncomment the following lines if you're using PyTorch
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_esm_embedding(model, batch_tokens, mean=False, return_contacts=False):
    '''
    Computes the ESM1b embedding for a given protein sequence and returns the mean embedding across positions as a single vector.

    Args:
        model (esm model): ESM1B model
        batch_tokens (tensor): sequences tensor
        mean (bool): should mean of the vector be returned

    Returns:
        torch.Tensor: A tensor representing the mean embedding across all positions in the protein sequence.
    '''
    model.eval()  # disables dropout for deterministic results
    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=return_contacts)
    token_representations = results["representations"][33]
    
    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    
    if mean:
        if token_representations.shape[0]==1:
            return token_representations[:,1:-1].mean(1)
        else:
            return token_representations[:,1:-1,:].mean(1)
    else:
        if token_representations.shape[0]==1:
            return token_representations[:,1:-1]
        else:
            return token_representations[:,1:-1,:]

def fasta_to_dataframe(fasta_file):
    """
    Converts a FASTA file into a pandas DataFrame

    Args:
    - fasta_file (str): Path to the FASTA file to be converted

    Returns:
    - pandas.DataFrame: DataFrame containing sequence IDs, sequences, HLA names and HLA lengths
    """
    records = SeqIO.parse(fasta_file, "fasta")
    data = [(record.id, str(record.seq), 
             record.description.split()[1], 
             record.description.split()[2]) for record in records]
    df = pd.DataFrame(data, columns=['ID', 'Sequence','HLA', 'length'])
    return df
