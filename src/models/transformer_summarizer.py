import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class TransformerSummarizer(nn.Module):
    """Transformer-based video summarization model."""
    
    def __init__(self, feature_dim: int, d_model: int = 512, nhead: int = 8, num_layers: int = 6):
        """
        Initialize the transformer summarizer.
        
        Args:
            feature_dim (int): Dimension of input features
            d_model (int): Dimension of transformer model
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer layers
        """
        super().__init__()
        
        # Feature projection
        self.feature_projection = nn.Linear(feature_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Score prediction
        self.score_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()  # Normalize scores to [0, 1]
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features (torch.Tensor): Input features of shape [batch_size, seq_len, feature_dim]
            
        Returns:
            torch.Tensor: Predicted importance scores of shape [batch_size, seq_len]
        """
        # Project features
        x = self.feature_projection(features)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer
        x = self.transformer_encoder(x)
        
        # Predict scores
        scores = self.score_predictor(x)
        
        return scores.squeeze(-1)  # Remove last dimension

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model (int): Dimension of model
            max_len (int): Maximum sequence length
        """
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            torch.Tensor: Input with positional encoding added
        """
        return x + self.pe[:, :x.size(1)] 