import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
import numpy as np
import os
import json

class VideoSummarizationDataset(Dataset):
    """Dataset class for video summarization."""
    
    def __init__(self, 
                 features_dir: str,
                 scores_file: Optional[str] = None,
                 transform: Optional[torch.nn.Module] = None):
        """
        Initialize the dataset.
        
        Args:
            features_dir (str): Directory containing pre-extracted features
            scores_file (str, optional): Path to JSON file containing ground truth scores
            transform (torch.nn.Module, optional): Transform to apply to features
        """
        self.features_dir = features_dir
        self.transform = transform
        
        # Load video IDs from features directory
        self.video_ids = [f for f in os.listdir(features_dir) 
                         if os.path.isdir(os.path.join(features_dir, f))]
        
        # Load ground truth scores if available
        self.scores = {}
        if scores_file and os.path.exists(scores_file):
            with open(scores_file, 'r') as f:
                self.scores = json.load(f)
    
    def __len__(self) -> int:
        """Return the number of videos in the dataset."""
        return len(self.video_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a video's features and scores (if available).
        
        Args:
            idx (int): Index of the video
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing features and scores
        """
        video_id = self.video_ids[idx]
        features_path = os.path.join(self.features_dir, video_id, "features.pt")
        
        # Load features
        features = torch.load(features_path)
        
        # Apply transform if specified
        if self.transform:
            features = self.transform(features)
        
        # Prepare output dictionary
        output = {
            "features": features,
            "video_id": video_id
        }
        
        # Add scores if available
        if video_id in self.scores:
            scores = torch.tensor(self.scores[video_id], dtype=torch.float32)
            output["scores"] = scores
        
        return output
    
    def get_video_ids(self) -> List[str]:
        """Return list of video IDs in the dataset."""
        return self.video_ids.copy()

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching variable-length sequences.
    
    Args:
        batch (List[Dict[str, torch.Tensor]]): List of dataset items
        
    Returns:
        Dict[str, torch.Tensor]: Batched features and scores
    """
    # Get maximum sequence length in the batch
    max_len = max(item["features"].size(0) for item in batch)
    
    # Initialize tensors for batched data
    batch_size = len(batch)
    feature_dim = batch[0]["features"].size(1)
    
    features = torch.zeros(batch_size, max_len, feature_dim)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    video_ids = []
    
    # Check if scores are available
    has_scores = "scores" in batch[0]
    if has_scores:
        scores = torch.zeros(batch_size, max_len)
    
    # Fill in the tensors
    for i, item in enumerate(batch):
        seq_len = item["features"].size(0)
        features[i, :seq_len] = item["features"]
        mask[i, :seq_len] = True
        video_ids.append(item["video_id"])
        
        if has_scores:
            scores[i, :seq_len] = item["scores"]
    
    # Prepare output dictionary
    output = {
        "features": features,
        "mask": mask,
        "video_ids": video_ids
    }
    
    if has_scores:
        output["scores"] = scores
    
    return output 