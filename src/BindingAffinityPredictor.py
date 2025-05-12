import pandas as pd
import numpy as np

from Bio import SeqIO
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchinfo import summary
import esm
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve

from utils import *
from utils_torch import * 
from MHCCBM import *

from tqdm import tqdm
import wandb


# Define the Protein Sequence Dataset
class ProteinSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        seq = self.X[idx]
        label = self.y[idx]
        return seq, label
    
class BindingAffinityPredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.esm_model, self.alphabet = None, None
        self.esm_batch_converter = None
        self.criterion = None
        self.optimizer = None
        
        layers = []
        self.in_features = input_size
        in_features = self.in_features
        
        #################
        for out_features in hidden_size:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.GELU())
#             layers.append(nn.Dropout(0.05))
            in_features = out_features
        layers.append(nn.Linear(in_features, 1))
        self.model = nn.Sequential(*layers)
        
#         ###################
#         for out_features in hidden_size:
#             layers.append(nn.LSTM(in_features, out_features, batch_first=True))
#             in_features = out_features
            
#         layers.append(nn.Linear(in_features, 1))
#         self.model = nn.Sequential(*layers)  
        
        ##################
#         layers.append(nn.LSTM(in_features, hidden_size[0], batch_first=True, num_layers=1))
#         in_features = hidden_size[0]
#         for out_features in hidden_size[1:]:
#             layers.append(nn.Linear(in_features, out_features))
#             layers.append(nn.GELU())
#             in_features = out_features
#         layers.append(nn.Linear(in_features, 1))
        
#         self.model = nn.Sequential(*layers)
        
        
    def forward(self, X):
        
        ###########
#         X, (h1, c1) = self.model[0](X)
#         output = self.model[1:](X)

#         ###########
#         for i in self.model[:-1]:
#             X, (h1, c1) = i(X)
            
#         output = self.model[-1](X)
        
        ######
        output = self.model(X)
        return output
         
    def embed_sequence(self, batch):
        data = [('seq'+str(idx), seq) for idx, seq in enumerate(batch)]
        batch_labels, batch_strs, batch_tokens = self.esm_batch_converter(data)
        
        embedding_rep = [None]*batch_tokens.shape[0]
        for idx, batch_token in tqdm(enumerate(batch_tokens)):
            embedding_rep[idx] = get_esm_embedding(model, batch_tokens[idx:idx+1])
        embedding_rep = torch.stack(embedding_rep) 
        
        return embedding_rep
    
    def embed_allele_peptide(self, allele_batch, peptide_batch):
        allele_rep = self.embed_sequence(allele_batch)
        peptide_rep = self.embed_sequence(peptide_batch)
        
        # Concatenate the representations
        combined_rep = torch.cat((allele_rep, peptide_rep), dim=2)
        return combined_rep
        
    def train_loop(self, train_loader, valid_loader, **kwargs):
        
        config_dict = kwargs.get('config_dict', {})
        config = config_dict['config']
        
        wandb.init(
            # set the wandb project where this run will be logged
            project = config_dict['project'],
            
            # set name for the run
            name = config_dict['name'],

            # track hyperparameters and run metadata
            config = config)
        

        # Loss and optimizer
#         self.criterion = nn.BCELoss()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=kwargs.get('pos_weight', [1]))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'], weight_decay=1e-3)
        nepochs = config['epochs']
#         clip_value = 0.0  # Set the value for gradient clipping
        
        ######### Training
        self.train()
        for epoch in range(nepochs):
            print("epoch: ", epoch)
            
            running_loss = 0.0
            device = get_device()
            
            for i, (batch, label_batch) in enumerate(train_loader):
                label_batch = label_batch.float().to(device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                output = self(batch)
                loss = self.criterion(output.squeeze(), label_batch)
                
                # Backward pass and optimization
                loss.backward()
                
                ## Gradient clipping
#                 torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value)
                
                self.optimizer.step()

                # Print statistics
                running_loss += loss.item()
                if i % 10 == 9:  # Print every 10 mini-batches
                    print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}")
                    running_loss = 0.0
                    
                    # log metrics to wandb
                    wandb.log({"loss": loss})
                    
            ######### Validation step
            self.eval()
            val_preds = []
            val_labels = []
            val_proba = []
            
            with torch.no_grad():
                for i, (batch, label_batch) in enumerate(valid_loader):
                    val_labels.extend(label_batch)
                    label_batch = label_batch.float().to(device)

                    # Forward pass
                    y_proba = self(batch)
                    y_pred = [1 if i>0.5 else 0 for i in y_proba.cpu().numpy()]
                    
                    val_proba.extend(y_proba)
                    val_preds.extend(y_pred)
                    

            val_f1 = f1_score(val_labels, val_preds, average='weighted')         
            
            fpr, tpr, _ = roc_curve(val_labels, val_proba)
            val_auroc = auc(fpr, tpr)

            precision, recall, _ = precision_recall_curve(val_labels, val_proba)
            val_auprc = auc(recall, precision)

            # Log validation F1 score, AUROC, and AUPRC to W&B
            wandb.log({
                "epoch": epoch + 1, 
                "val_f1": val_f1,
                "val_auroc": val_auroc,
                "val_auprc": val_auprc})
            
        summary(self, input_size=(config['batch_size'],1, self.in_features))
        
        # Finalize W&B run
        wandb.finish() 
    
    
    def eval_dataset(self, dataloader, return_prob=False, return_label=False):
        
        self.eval()
        preds_ls = []
        labels_ls = []
        proba_ls = []
        device = get_device()
        
        with torch.no_grad():
            for i, (batch, label_batch) in enumerate(dataloader):
                labels_ls.extend(label_batch)
                label_batch = label_batch.float().to(device)

                # Forward pass
                y_proba = self(batch)
                sig = nn.Sigmoid()
                y_pred = [1 if sig(i).cpu().numpy()[0]>0.5 else 0 for i in y_proba]

                proba_ls.extend(y_proba)
                preds_ls.extend(y_pred)


        f1, auroc_score, auprc_score = self.get_metrics(labels_ls, preds_ls, proba_ls)
        
        if return_prob and return_label:
            return f1, auroc_score, auprc_score, proba_ls, preds_ls
        elif return_prob==False and return_label:
            return f1, auroc_score, auprc_score, preds_ls
        elif return_prob and return_label==False:
            return f1, auroc_score, auprc_score, proba_ls
        else:
            return f1, auroc_score, auprc_score
    
    
    def get_metrics(self, labels_ls, preds_ls, proba_ls):
        f1 = f1_score(labels_ls, preds_ls, average='weighted')
        
        fpr, tpr, _ = roc_curve(labels_ls, proba_ls)
        auroc_score = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(labels_ls, proba_ls)
        auprc_score = auc(recall, precision)
        
        return f1, auroc_score, auprc_score 
    