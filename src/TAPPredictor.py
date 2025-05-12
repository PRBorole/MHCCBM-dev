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
    
class TAPPredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.esm_model, self.alphabet = None, None
        self.esm_batch_converter = None
        self.criterion = None
        self.optimizer = None
        
        layers = []
        self.in_features = input_size
        in_features = self.in_features
        
        for out_features in hidden_size:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.Sigmoid())
            layers.append(nn.Dropout(0.01))
            in_features = out_features
        layers.append(nn.Linear(in_features, 1))
        self.model = nn.Sequential(*layers)
        
    def forward(self, X):
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
        
    def train_loop(self, train_loader, valid_loader, test_loader, **kwargs):
        
        config_dict = kwargs.get('config_dict', {})
        config = config_dict['config']
        
#         wandb.init(
#             # set the wandb project where this run will be logged
#             project = config_dict['project'],
            
#             # set name for the run
#             name = config_dict['name'],

#             # track hyperparameters and run metadata
#             config = config)
        

        # Loss and optimizer
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=kwargs.get('pos_weight', [1]))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        nepochs = config['epochs']
        
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
                
                self.optimizer.step()

                # Print statistics
                running_loss += loss.item()
                if i % 10 == 9:  # Print every 10 mini-batches
                    print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}")
                    running_loss = 0.0
                    
#                     # log metrics to wandb
#                     wandb.log({"loss": loss})
            
            ######### Validation step
            if valid_loader is not None:
                
                self.eval()
                val_preds = []
                val_labels = []
                val_proba = []

                with torch.no_grad():
                    for i, (batch, label_batch) in enumerate(valid_loader):
                        val_labels.extend(label_batch)
                        label_batch = label_batch.float().to(device)

                        # Forward pass
                        y_proba = torch.sigmoid(self(batch))
                        y_pred = torch.where(y_proba>0.5,1,0)

                        val_proba.extend(y_proba)
                        val_preds.extend(y_pred)

                val_proba = torch.cat(val_proba).numpy().reshape(-1)
                val_preds = torch.cat(val_preds).numpy().reshape(-1)

                val_f1, val_auroc, val_auprc = self.get_metrics(val_labels, val_preds, val_proba)
                print(val_f1, val_auroc, val_auprc)
#                 wandb.log({"epoch": epoch + 1,
#                             "val_f1": val_f1,
#                             "val_auroc": val_auroc,
#                             "val_auprc": val_auprc})
                
            ######### test step
            if test_loader is not None:
                self.eval()
                test_preds = []
                test_labels = []
                test_proba = []

                with torch.no_grad():
                    for i, (batch, label_batch) in enumerate(test_loader):
                        test_labels.extend(label_batch)
                        label_batch = label_batch.float().to(device)

                        # Forward pass
                        y_proba = torch.sigmoid(self(batch))
                        y_pred = torch.where(y_proba>0.5,1,0)

                        test_proba.extend(y_proba)
                        test_preds.extend(y_pred)

                test_proba = torch.cat(test_proba).numpy().reshape(-1)
                test_preds = torch.cat(test_preds).numpy().reshape(-1)

                test_f1, test_auroc, test_auprc = self.get_metrics(test_labels, test_preds, test_proba)

                # Log validation F1 score, AUROC, and AUPRC to W&B
#                 wandb.log({"epoch": epoch + 1,
#                             "test_f1": test_f1,
#                             "test_auroc": test_auroc,
#                             "test_auprc": test_auprc})
            
        summary(self, input_size=(config['batch_size'], self.in_features))
        
#         # Finalize W&B run
#         wandb.finish() 
    
    
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
                y_proba = torch.sigmoid(self(batch))
                y_pred = torch.where(y_proba>0.5,1,0)

                proba_ls.extend(y_proba)
                preds_ls.extend(y_pred)

        proba_ls = torch.cat(proba_ls).numpy().reshape(-1)
        preds_ls = torch.cat(preds_ls).numpy().reshape(-1)
        
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
    