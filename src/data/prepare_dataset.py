import os
import json
import torch
import torchvision.models as models
import torchvision.transforms
from tqdm import tqdm
import cv2
import numpy as np
from typing import List, Dict
from pathlib import Path

class DatasetPreparator:
    """Class for preparing the video summarization dataset."""
    
    def __init__(self,
                 video_dir: str,
                 output_dir: str,
                 frame_rate: int = 1):
        """
        Initialize the dataset preparator.
        
        Args:
            video_dir (str): Directory containing input videos
            output_dir (str): Directory to save processed data
            frame_rate (int): Frame sampling rate
        """
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.frame_rate = frame_rate
        
        # Create output directories
        self.features_dir = self.output_dir / "features"
        self.frames_dir = self.output_dir / "frames"
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup feature extractor
        self.feature_extractor = self._setup_feature_extractor()
    
    def _setup_feature_extractor(self) -> torch.nn.Module:
        """Setup the pre-trained ResNet50 model for feature extraction."""
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
        return model
    
    def process_video(self, video_path: Path) -> Dict:
        """
        Process a single video file.
        
        Args:
            video_path (Path): Path to the video file
            
        Returns:
            Dict: Dictionary containing video information and features
        """
        # Create output directories for this video
        video_id = video_path.stem
        video_features_dir = self.features_dir / video_id
        video_frames_dir = self.frames_dir / video_id
        video_features_dir.mkdir(exist_ok=True)
        video_frames_dir.mkdir(exist_ok=True)
        
        # Load video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        frame_count = 0
        
        # Sample frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % int(fps / self.frame_rate) == 0:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                
                # Save frame
                frame_path = video_frames_dir / f"frame_{len(frames):04d}.jpg"
                cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            frame_count += 1
        
        cap.release()
        
        # Extract features
        features = self._extract_features(frames)
        
        # Save features
        features_path = video_features_dir / "features.pt"
        torch.save(features, features_path)
        
        # Generate dummy importance scores (for demonstration)
        scores = self._generate_dummy_scores(len(frames))
        
        return {
            "video_id": video_id,
            "num_frames": len(frames),
            "fps": fps,
            "total_frames": total_frames,
            "scores": scores
        }
    
    def _extract_features(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Extract features from frames using ResNet50.
        
        Args:
            frames (List[np.ndarray]): List of frames
            
        Returns:
            torch.Tensor: Tensor of features
        """
        # Preprocess frames
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Process frames in batches
        batch_size = 32
        features_list = []
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_tensors = torch.stack([transform(frame) for frame in batch_frames])
            
            with torch.no_grad():
                batch_features = self.feature_extractor(batch_tensors)
                features_list.append(batch_features.squeeze())
        
        return torch.cat(features_list)
    
    def _generate_dummy_scores(self, num_frames: int) -> List[float]:
        """
        Generate dummy importance scores for demonstration.
        
        Args:
            num_frames (int): Number of frames
            
        Returns:
            List[float]: List of importance scores
        """
        # Generate random scores with some temporal smoothness
        scores = np.random.randn(num_frames)
        scores = np.convolve(scores, np.ones(5)/5, mode='same')
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        return scores.tolist()
    
    def prepare_dataset(self):
        """Process all videos in the input directory."""
        video_files = list(self.video_dir.glob("*.mp4"))
        dataset_info = {}
        
        for video_path in tqdm(video_files, desc="Processing videos"):
            info = self.process_video(video_path)
            dataset_info[info["video_id"]] = info
        
        # Save dataset information
        info_path = self.output_dir / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # Save scores separately
        scores = {video_id: info["scores"] for video_id, info in dataset_info.items()}
        scores_path = self.output_dir / "scores.json"
        with open(scores_path, 'w') as f:
            json.dump(scores, f, indent=2)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare video summarization dataset")
    parser.add_argument("--video_dir", required=True,
                      help="Directory containing input videos")
    parser.add_argument("--output_dir", required=True,
                      help="Directory to save processed data")
    parser.add_argument("--frame_rate", type=int, default=1,
                      help="Frame sampling rate")
    
    args = parser.parse_args()
    
    preparator = DatasetPreparator(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        frame_rate=args.frame_rate
    )
    
    preparator.prepare_dataset()

if __name__ == "__main__":
    main() 