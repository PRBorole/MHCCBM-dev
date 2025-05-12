import pandas as pd
import numpy as np
from Bio import SeqIO

def fasta_to_dataframe(fasta_file):
    """
    Converts a FASTA file into a pandas DataFrame

    Parameters:
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