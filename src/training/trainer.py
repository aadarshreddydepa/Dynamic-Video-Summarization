import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime

class VideoSummarizationTrainer:
    """Trainer class for video summarization model."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 criterion: Optional[nn.Module] = None,
                 device: Optional[torch.device] = None,
                 checkpoint_dir: str = "models"):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): The model to train
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader, optional): Validation data loader
            optimizer (torch.optim.Optimizer, optional): Optimizer
            criterion (nn.Module, optional): Loss function
            device (torch.device, optional): Device to train on
            checkpoint_dir (str): Directory to save checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize optimizer and criterion if not provided
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=1e-4)
        self.criterion = criterion or nn.MSELoss()
        
        # Setup checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": []
        }
    
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Returns:
            Tuple[float, Dict[str, float]]: Average loss and metrics
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch in tqdm(self.train_loader, desc="Training"):
            # Move data to device
            features = batch["features"].to(self.device)
            mask = batch["mask"].to(self.device)
            targets = batch["scores"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            preds = self.model(features, mask)
            
            # Compute loss
            loss = self.criterion(preds, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            all_preds.extend(preds[mask].cpu().numpy())
            all_targets.extend(targets[mask].cpu().numpy())
        
        # Compute average loss and metrics
        avg_loss = total_loss / len(self.train_loader)
        metrics = self._compute_metrics(np.array(all_preds), np.array(all_targets))
        
        return avg_loss, metrics
    
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model.
        
        Returns:
            Tuple[float, Dict[str, float]]: Average loss and metrics
        """
        if not self.val_loader:
            return 0.0, {}
        
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move data to device
                features = batch["features"].to(self.device)
                mask = batch["mask"].to(self.device)
                targets = batch["scores"].to(self.device)
                
                # Forward pass
                preds = self.model(features, mask)
                
                # Compute loss
                loss = self.criterion(preds, targets)
                
                # Update metrics
                total_loss += loss.item()
                all_preds.extend(preds[mask].cpu().numpy())
                all_targets.extend(targets[mask].cpu().numpy())
        
        # Compute average loss and metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = self._compute_metrics(np.array(all_preds), np.array(all_targets))
        
        return avg_loss, metrics
    
    def train(self, num_epochs: int, save_best: bool = True) -> Dict[str, list]:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs (int): Number of epochs to train
            save_best (bool): Whether to save the best model
            
        Returns:
            Dict[str, list]: Training history
        """
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train and validate
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate()
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_metrics"].append(train_metrics)
            self.history["val_metrics"].append(val_metrics)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print("Train Metrics:", train_metrics)
            print("Val Metrics:", val_metrics)
            
            # Save best model
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(f"best_model.pt")
            
            # Save latest checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """
        Save a model checkpoint.
        
        Args:
            filename (str): Name of the checkpoint file
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "timestamp": datetime.now().isoformat()
        }
        
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filename: str):
        """
        Load a model checkpoint.
        
        Args:
            filename (str): Name of the checkpoint file
        """
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
    
    def _compute_metrics(self, preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            preds (np.ndarray): Predicted scores
            targets (np.ndarray): Target scores
            
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        # Convert to binary predictions using threshold
        threshold = np.mean(targets)
        pred_binary = (preds > threshold).astype(int)
        target_binary = (targets > threshold).astype(int)
        
        # Compute metrics
        tp = np.sum((pred_binary == 1) & (target_binary == 1))
        fp = np.sum((pred_binary == 1) & (target_binary == 0))
        fn = np.sum((pred_binary == 0) & (target_binary == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        } 