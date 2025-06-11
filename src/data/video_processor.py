import cv2
import numpy as np
from typing import List, Tuple, Optional
import torch
from torchvision import transforms
from PIL import Image

class VideoProcessor:
    """A class for processing video files and extracting frames."""
    
    def __init__(self, frame_rate: int = 1):
        """
        Initialize the VideoProcessor.
        
        Args:
            frame_rate (int): Number of frames to sample per second
        """
        self.frame_rate = frame_rate
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def load_video(self, video_path: str) -> Tuple[List[np.ndarray], float]:
        """
        Load a video file and sample frames at the specified frame rate.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            Tuple[List[np.ndarray], float]: List of frames and video FPS
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % int(fps / self.frame_rate) == 0:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        return frames, fps
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single frame for feature extraction.
        
        Args:
            frame (np.ndarray): Input frame as numpy array
            
        Returns:
            torch.Tensor: Preprocessed frame as tensor
        """
        # Convert numpy array to PIL Image
        frame_pil = Image.fromarray(frame)
        # Apply transforms
        frame_tensor = self.transform(frame_pil)
        return frame_tensor.unsqueeze(0)  # Add batch dimension
    
    def extract_features(self, frames: List[np.ndarray], 
                        model: torch.nn.Module,
                        device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Extract features from a list of frames using the provided model.
        
        Args:
            frames (List[np.ndarray]): List of frames
            model (torch.nn.Module): Pre-trained model for feature extraction
            device (torch.device, optional): Device to run the model on
            
        Returns:
            torch.Tensor: Tensor of shape (num_frames, feature_dim)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = model.to(device)
        model.eval()
        
        features = []
        with torch.no_grad():
            for frame in frames:
                frame_tensor = self.preprocess_frame(frame).to(device)
                feature = model(frame_tensor)
                features.append(feature.squeeze().cpu())
        
        return torch.stack(features)
    
    def save_frames(self, frames: List[np.ndarray], 
                   output_dir: str, 
                   prefix: str = "frame") -> None:
        """
        Save frames to disk.
        
        Args:
            frames (List[np.ndarray]): List of frames to save
            output_dir (str): Directory to save frames
            prefix (str): Prefix for frame filenames
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for i, frame in enumerate(frames):
            frame_path = os.path.join(output_dir, f"{prefix}_{i:04d}.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) 