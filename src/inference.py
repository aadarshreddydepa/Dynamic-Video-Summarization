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
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error

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
    
    def calculate_metrics(self, selected_indices: List[int], total_frames: int, ground_truth_indices: List[int] = None) -> dict:
        """
        Calculate evaluation metrics.
        
        Args:
            selected_indices (List[int]): Indices of selected frames
            total_frames (int): Total number of frames in video
            ground_truth_indices (List[int], optional): Ground truth frame indices
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Create binary arrays for selected frames
        selected_binary = np.zeros(total_frames)
        selected_binary[selected_indices] = 1
        
        # If ground truth is not provided, use a simple heuristic
        # (frames with scores above mean are considered important)
        if ground_truth_indices is None:
            # Use a simple heuristic: frames in the middle third are considered important
            middle_start = total_frames // 3
            middle_end = (2 * total_frames) // 3
            ground_truth_indices = list(range(middle_start, middle_end))
        
        ground_truth_binary = np.zeros(total_frames)
        ground_truth_binary[ground_truth_indices] = 1
        
        # Calculate metrics
        precision = precision_score(ground_truth_binary, selected_binary)
        recall = recall_score(ground_truth_binary, selected_binary)
        f1 = f1_score(ground_truth_binary, selected_binary)
        mse = mean_squared_error(ground_truth_binary, selected_binary)
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'mse': float(mse)
        }
    
    def generate_summary(self,
                        video_path: str,
                        output_dir: str,
                        num_frames: int = 100,
                        chunk_size: int = 100) -> Tuple[List[np.ndarray], List[float], dict]:
        """
        Generate a video summary.
        
        Args:
            video_path (str): Path to input video
            output_dir (str): Directory to save summary frames
            num_frames (int): Number of frames to include in summary
            chunk_size (int): Number of frames to process at once
            
        Returns:
            Tuple[List[np.ndarray], List[float], dict]: Selected frames, their scores, and evaluation metrics
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
        
        # Save scores for all frames
        all_scores_path = output_dir / 'all_frame_scores.json'
        with open(all_scores_path, 'w') as f:
            json.dump({
                'frame_scores': [float(score) for score in all_scores],
                'total_frames': len(all_scores),
                'average_score': float(np.mean(all_scores)),
                'min_score': float(np.min(all_scores)),
                'max_score': float(np.max(all_scores))
            }, f, indent=2)
        print(f"\nAll frame scores saved to: {all_scores_path}")
        
        # Select top frames
        top_indices = np.argsort(all_scores)[-num_frames:]
        selected_frames = [all_frames[i] for i in top_indices]
        selected_scores = [all_scores[i] for i in top_indices]
        
        # Calculate evaluation metrics
        metrics = self.calculate_metrics(top_indices, len(all_frames))
        
        # Save metrics
        metrics_path = output_dir / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save frames
        for i, (frame, score) in enumerate(zip(selected_frames, selected_scores)):
            frame_path = output_dir / f'frame_{i:04d}_score_{score:.4f}.jpg'
            cv2.imwrite(str(frame_path), frame)
        
        # Save scores for selected frames
        selected_scores_path = output_dir / 'selected_frame_scores.json'
        with open(selected_scores_path, 'w') as f:
            json.dump({
                'frame_scores': [float(score) for score in selected_scores],
                'frame_indices': [int(idx) for idx in top_indices],
                'num_frames': len(selected_scores),
                'average_score': float(np.mean(selected_scores)),
                'min_score': float(np.min(selected_scores)),
                'max_score': float(np.max(selected_scores))
            }, f, indent=2)
        print(f"Selected frame scores saved to: {selected_scores_path}")
            
        # Create video from selected frames
        print("\nCreating summary video...")
        video_output_path = output_dir / "summary_video.mp4"
        height, width = selected_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Set FPS to 10 for a 10-second video (100 frames / 10 seconds = 10 FPS)
        fps = 10
        out = cv2.VideoWriter(str(video_output_path), fourcc, fps, (width, height))
        
        for frame in selected_frames:
            out.write(frame)
        out.release()
        print(f"Summary video saved to: {video_output_path}")
        
        return selected_frames, selected_scores, metrics

def main():
    video_path1 = input("Enter the name of the video: ")
    # Set paths
    model_path = 'models/best_model.pt'
    video_path = f"data/videos/{video_path1}"  # Removed extra .mp4
    
    # Create output directory based on video name (without extension)
    video_name = os.path.splitext(video_path1)[0]
    output_dir = f'outputs/summaries/{video_name}'  # Create separate directory for each video
    
    print("Initializing video summarizer...")
    summarizer = VideoSummarizer(model_path)
    
    print(f"\nProcessing video: {video_path}")
    frames, scores, metrics = summarizer.generate_summary(
        video_path=video_path,
        output_dir=output_dir,
        num_frames=100  # Default number of frames
    )
    
    print(f"\nSummary generated successfully!")
    print(f"Selected {len(frames)} frames")
    print(f"Average importance score: {np.mean(scores):.4f}")
    print(f"\nEvaluation Metrics:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"\nOutput saved to: {output_dir}")

if __name__ == '__main__':
    main() 