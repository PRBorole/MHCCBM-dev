import pandas as pd
import numpy as np
from tqdm import tqdm

from Bio import SeqIO
import torch
from torch import nn
from torch.utils.data import DataLoader
import esm

from utils import *

        
def get_device():
    device = ("cuda"
              if torch.cuda.is_available()
              else "mps"
              if torch.backends.mps.is_available()
              else "cpu")
    return device
    
    
        