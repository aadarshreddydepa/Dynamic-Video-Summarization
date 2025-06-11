import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path
import cv2
import numpy as np
import json
from tqdm import tqdm

def setup_feature_extractor():
    """Setup the pre-trained ResNet50 model for feature extraction."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

def process_video(video_path: str, output_dir: str, frame_rate: int = 1):
    """
    Process a video file to extract features and generate scores.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save processed data
        frame_rate (int): Frame sampling rate
    """
    # Create output directories
    output_dir = Path(output_dir)
    features_dir = output_dir / "features"
    frames_dir = output_dir / "frames"
    features_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup feature extractor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = setup_feature_extractor().to(device)
    
    # Setup image transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    frame_count = 0
    
    print("Extracting frames...")
    # Sample frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % int(fps / frame_rate) == 0:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
            # Save frame
            frame_path = frames_dir / f"frame_{len(frames):04d}.jpg"
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        frame_count += 1
    
    cap.release()
    
    print("Extracting features...")
    # Extract features
    features_list = []
    batch_size = 32
    
    for i in tqdm(range(0, len(frames), batch_size)):
        batch_frames = frames[i:i + batch_size]
        batch_tensors = torch.stack([transform(frame) for frame in batch_frames]).to(device)
        
        with torch.no_grad():
            batch_features = feature_extractor(batch_tensors)
            features_list.append(batch_features.squeeze().cpu())
    
    features = torch.cat(features_list)
    
    # Generate dummy importance scores (for demonstration)
    scores = np.random.randn(len(frames))
    scores = np.convolve(scores, np.ones(5)/5, mode='same')  # Add temporal smoothness
    scores = (scores - scores.min()) / (scores.max() - scores.min())  # Normalize to [0, 1]
    
    # Save features
    video_id = Path(video_path).stem
    video_features_dir = features_dir / video_id
    video_features_dir.mkdir(exist_ok=True)
    torch.save(features, video_features_dir / "features.pt")
    
    # Save scores
    scores_dict = {video_id: scores.tolist()}
    with open(output_dir / "scores.json", 'w') as f:
        json.dump(scores_dict, f, indent=2)
    
    print(f"Processed {len(frames)} frames")
    print(f"Features saved to: {video_features_dir}")
    print(f"Scores saved to: {output_dir / 'scores.json'}")

def main():
    # Set paths
    video_path = "data/videos/sample.mp4"
    output_dir = "data/processed"
    
    # Process video
    process_video(video_path, output_dir, frame_rate=1)

if __name__ == "__main__":
    main() 