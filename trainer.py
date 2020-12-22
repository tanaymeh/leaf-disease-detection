import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm

from sklearn.metrics import accuracy_score

class Trainer:
    def __init__(self, train_dataloader, valid_dataloader, model, optimizer, loss_fn, val_loss_fn, scheduler, device="cuda:0", plot_results=True):
        """
        TODO: Implement the ROC-AUC Scheduler stuff
        """
        self.train = train_dataloader
        self.valid = valid_dataloader
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn
        self.val_loss_fn = val_loss_fn
        self.scheduler = scheduler
        self.device = device
        self.plot_results = plot_results
    
    def train_one_cycle(self):
        """
        Runs one epoch of training, backpropagation, optimization and gets train accuracy
        """
        print(f'[TRAINING]')

        self.model.train()
        train_prog_bar = tqdm(self.train, total=len(self.train))

        all_train_labels = []
        all_train_preds = []
        
        running_loss = 0
        
        for x, y in train_prog_bar:
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            y = torch.tensor(y, device=self.device, dtype=torch.long)
            
            # Get predictions
            z = self.model(x)
            
            # Training
            train_loss = self.loss_fn(z, y)
            train_loss.backward()
            self.optim.step()

            # For averaging and reporting later
            running_loss += train_loss
            
            # Convert the predictions and corresponding labels to right form
            train_predictions = z.argmax(axis=1).cpu().detach().numpy()
            train_labels = y.cpu().detach().numpy()
        
            # Append current predictions and current labels to a list
            all_train_labels += train_predictions.tolist()
            all_train_preds += all_train_preds.tolist()

            # Show the current loss to the progress bar
            train_pbar_desc = f'loss: {train_loss.item():.4f}'
            train_prog_bar.set_description(desc=train_pbar_desc)
        
        # After all the batches are done, calculate the training accuracy
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        print(f"Training Accuracy: {train_acc:.4f}")
        
        # Now average the running loss over all batches and return
        train_running_loss = running_loss / len(self.train)
        
        return (train_acc, train_running_loss)

    def valid_one_cycle(self):
        """
        Runs one epoch of prediction and validation accuracy calculation
        """
        print(f'[VALIDATING]')
        
        self.model.eval()
        
        valid_prog_bar = tqdm(self.valid, total=len(self.valid))
        
        with torch.no_grad():
            all_valid_labels = []
            all_valid_preds = []
            
            running_loss = 0
            
            for x, y in valid_prog_bar:
                x = torch.tensor(x, device=self.device, dtype=torch.float32)
                y = torch.tensor(y, device=self.device, dtype=torch.long)
                
                val_z = self.model(x)
                
                val_loss = self.val_loss_fn(val_z, y)
                
                running_loss += val_loss.item()
                
                val_pred = z_val.argmax(axis=1).cpu().detach().numpy()
                val_label = y.cpu().detach().numpy()
                
                all_valid_labels += val_label.tolist()
                all_valid_preds += val_pred.tolist()
            
                # Show the current loss
                valid_pbar_desc = f"loss: {running_loss.item():.4f}"
                valid_prog_bar.set_description(desc=valid_pbar_desc)
            
            # Get the final loss
            final_loss_val = running_loss / len(self.valid)
            
            # TODO: Complete the Trainer Class