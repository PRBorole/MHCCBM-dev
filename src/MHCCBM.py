import pandas as pd
import numpy as np
from tqdm import tqdm

from Bio import SeqIO
import torch
from torch import nn
from torch.utils.data import DataLoader
import esm

from utils import *
from utils_torch import * 

class MHCCBM():
    
    def __init__(self):
        super().__init__()
        sself.esm_model, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Freeze the ESM model parameters
        for param in self.esm_model.parameters():
            param.requires_grad = False
        
    def get_esm_embedding(self, sequences):
        # Convert sequences to the format required by ESM
        batch_labels, batch_strs, batch_tokens = self.batch_converter(sequences)
        batch_tokens = batch_tokens.to(next(self.esm_model.parameters()).device)
        

        # Get the embeddings from ESM-1b
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
        
        # Extract embeddings
        token_representations = results['representations'][33]
        
        return token_representations

        
    