import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from models.transformer_summarizer import TransformerSummarizer

class VideoSummarizationDataset(Dataset):
    """Dataset for video summarization."""
    
    def __init__(self, features_dir: str, scores_file: str):
        """
        Initialize the dataset.
        
        Args:
            features_dir (str): Directory containing feature files
            scores_file (str): Path to scores JSON file
        """
        self.features_dir = Path(features_dir)
        with open(scores_file, 'r') as f:
            self.scores = json.load(f)
        
        self.video_ids = list(self.scores.keys())
    
    def __len__(self) -> int:
        return len(self.video_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_id = self.video_ids[idx]
        
        # Load features
        features_path = self.features_dir / video_id / "features.pt"
        features = torch.load(features_path)
        
        # Load scores
        scores = torch.tensor(self.scores[video_id], dtype=torch.float32)
        
        return features, scores

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                num_epochs: int,
                learning_rate: float,
                device: torch.device,
                patience: int = 5) -> Dict[str, List[float]]:
    """
    Train the model with early stopping.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Maximum number of epochs to train
        learning_rate (float): Learning rate
        device (torch.device): Device to train on
        patience (int): Number of epochs to wait for improvement before stopping
        
    Returns:
        Dict[str, List[float]]: Training history
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for features, scores in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            features = features.to(device)
            scores = scores.to(device)
            
            optimizer.zero_grad()
            pred_scores = model(features)
            loss = criterion(pred_scores, scores)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, scores in val_loader:
                features = features.to(device)
                scores = scores.to(device)
                
                pred_scores = model(features)
                loss = criterion(pred_scores, scores)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'models/best_model.pt')
        else:
            patience_counter += 1
            print(f'  No improvement for {patience_counter} epochs')
            
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model
    feature_dim = 2048  # ResNet50 feature dimension
    model = TransformerSummarizer(
        feature_dim=feature_dim,
        d_model=512,
        nhead=8,
        num_layers=6
    ).to(device)
    
    # Create datasets
    data_dir = Path('data/processed')
    dataset = VideoSummarizationDataset(
        features_dir=str(data_dir / 'features'),
        scores_file=str(data_dir / 'scores.json')
    )
    
    print(f'Total number of videos: {len(dataset)}')
    
    # For single video, use the same data for both training and validation
    if len(dataset) == 1:
        train_dataset = dataset
        val_dataset = dataset
    else:
        # Split into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    )
    
    # Train model with early stopping
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,  # Reduced from 50 to 20
        learning_rate=0.0001,
        device=device,
        patience=5  # Stop if no improvement for 5 epochs
    )
    
    # Save training history
    with open('models/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == '__main__':
    main() 