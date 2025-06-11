import os
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import json
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms

from models.transformer_summarizer import TransformerSummarizer

class VideoSummarizer:
    """Class for generating video summaries using the trained model."""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize the video summarizer.
        
        Args:
            model_path (str): Path to the trained model
            device (str): Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        # Load feature extractor
        self.feature_extractor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.feature_dim = 2048
        
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        # Remove the last layer to get features
        self.feature_extractor = torch.nn.Sequential(*list(self.feature_extractor.children())[:-1])
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load transformer model
        self.model = TransformerSummarizer(
            feature_dim=self.feature_dim,
            d_model=512,
            nhead=8,
            num_layers=6
        ).to(self.device)
        
        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def extract_features(self, frame: np.ndarray) -> torch.Tensor:
        """
        Extract features from a single frame.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            torch.Tensor: Extracted features
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess image
        img_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(img_tensor)
            features = features.squeeze()
        
        return features
    
    def predict_importance(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict importance scores for frames.
        
        Args:
            features (torch.Tensor): Frame features
            
        Returns:
            torch.Tensor: Predicted importance scores
        """
        with torch.no_grad():
            # Add batch dimension if needed
            if features.dim() == 1:
                features = features.unsqueeze(0)
            
            # Add sequence dimension if needed
            if features.dim() == 2:
                features = features.unsqueeze(0)
            
            # Predict scores
            scores = self.model(features)
            scores = scores.squeeze()  # Remove batch dimension
            
            if scores.dim() == 0:  # If only one score
                scores = scores.unsqueeze(0)
        
        return scores
    
    def process_chunk(self, frames: List[np.ndarray]) -> Tuple[List[torch.Tensor], List[np.ndarray]]:
        """
        Process a chunk of frames to extract features.
        
        Args:
            frames (List[np.ndarray]): List of frames to process
            
        Returns:
            Tuple[List[torch.Tensor], List[np.ndarray]]: Features and original frames
        """
        features_list = []
        processed_frames = []
        
        for frame in frames:
            features = self.extract_features(frame)
            features_list.append(features)
            processed_frames.append(frame)
        
        return features_list, processed_frames
    
    def generate_summary(self,
                        video_path: str,
                        output_dir: str,
                        num_frames: int = 10,
                        chunk_size: int = 100) -> Tuple[List[np.ndarray], List[float]]:
        """
        Generate a video summary.
        
        Args:
            video_path (str): Path to input video
            output_dir (str): Directory to save summary frames
            num_frames (int): Number of frames to include in summary
            chunk_size (int): Number of frames to process at once
            
        Returns:
            Tuple[List[np.ndarray], List[float]]: Selected frames and their scores
        """
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Total frames in video: {total_frames}')
        
        # Process video in chunks
        print('Processing video in chunks...')
        all_frames = []
        all_scores = []
        
        for start_idx in tqdm(range(0, total_frames, chunk_size)):
            # Read chunk of frames
            frames_chunk = []
            for _ in range(chunk_size):
                if start_idx + _ >= total_frames:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                frames_chunk.append(frame)
            
            if not frames_chunk:
                break
            
            # Process chunk
            features_chunk, frames_chunk = self.process_chunk(frames_chunk)
            
            # Predict scores for chunk
            features_tensor = torch.stack(features_chunk)
            scores_chunk = self.predict_importance(features_tensor)
            scores_chunk = scores_chunk.detach().cpu().numpy()
            
            # Store results
            all_frames.extend(frames_chunk)
            all_scores.extend(scores_chunk)
        
        cap.release()
        
        # Select top frames
        top_indices = np.argsort(all_scores)[-num_frames:]
        selected_frames = [all_frames[i] for i in top_indices]
        selected_scores = [all_scores[i] for i in top_indices]
        
        # Save frames
        for i, (frame, score) in enumerate(zip(selected_frames, selected_scores)):
            frame_path = output_dir / f'frame_{i:04d}_score_{score:.4f}.jpg'
            cv2.imwrite(str(frame_path), frame)
        
        # Save scores
        scores_path = output_dir / 'frame_scores.json'
        with open(scores_path, 'w') as f:
            json.dump({
                'frame_scores': [float(score) for score in selected_scores],
                'frame_indices': [int(idx) for idx in top_indices]
            }, f, indent=2)
        
        return selected_frames, selected_scores

def main():
    video_path1 = input("Enter the name of the video: ")
    # Set paths
    model_path = 'models/best_model.pt'
    video_path = f"data/videos/{video_path1}"  # Removed extra .mp4
    output_dir = 'data/summaries/sample'  # Regular sample directory
    
    print("Initializing video summarizer...")
    summarizer = VideoSummarizer(model_path)
    
    print(f"\nProcessing video: {video_path}")
    frames, scores = summarizer.generate_summary(
        video_path=video_path,
        output_dir=output_dir,
        num_frames=10  # Default number of frames
    )
    
    print(f"\nSummary generated successfully!")
    print(f"Selected {len(frames)} frames")
    print(f"Average importance score: {np.mean(scores):.4f}")
    print(f"Output saved to: {output_dir}")

if __name__ == '__main__':
    main() 