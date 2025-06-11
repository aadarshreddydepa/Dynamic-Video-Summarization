import torch
import torchvision.models as models
from torch.utils.data import DataLoader
import argparse
import os
from typing import Optional, List, Tuple
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
import sys
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import VideoSummarizationDataset, collate_fn
from models.transformer_summarizer import TransformerSummarizer
from training.trainer import VideoSummarizationTrainer
from inference import VideoSummarizer

def setup_feature_extractor() -> torch.nn.Module:
    """Setup the pre-trained ResNet50 model for feature extraction."""
    model = models.resnet50(pretrained=True)
    # Remove the final classification layer
    model = torch.nn.Sequential(*list(model.children())[:-1])
    return model

def select_video_file() -> str:
    """Open file dialog to select a video file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )
    return file_path

def create_output_directories(video_name: str) -> dict:
    """Create necessary directories for storing outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("outputs", f"{video_name}_{timestamp}")
    
    dirs = {
        "base": base_dir,
        "frames": os.path.join(base_dir, "frames"),
        "summary": os.path.join(base_dir, "summary"),
        "metrics": os.path.join(base_dir, "metrics"),
        "visualizations": os.path.join(base_dir, "visualizations")
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def calculate_metrics(scores: List[float], selected_indices: List[int], total_frames: int) -> dict:
    """Calculate comprehensive metrics for the summarization."""
    # Create binary labels (1 for selected frames, 0 for others)
    y_true = np.zeros(total_frames)
    y_true[selected_indices] = 1
    
    # Create predicted scores
    y_pred = np.zeros(total_frames)
    y_pred[selected_indices] = scores
    
    # Calculate metrics
    # Convert scores to binary predictions using a threshold
    threshold = np.mean(scores)
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate classification metrics
    precision = precision_score(y_true, y_pred_binary, average='binary')
    recall = recall_score(y_true, y_pred_binary, average='binary')
    f1 = f1_score(y_true, y_pred_binary, average='binary')
    
    # Calculate regression metrics
    mse = mean_squared_error(y_true, y_pred)
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "mse": float(mse),
        "selected_frames": len(selected_indices),
        "total_frames": total_frames,
        "compression_ratio": float(len(selected_indices)) / total_frames
    }

def visualize_metrics(scores: List[float], selected_indices: List[int], output_dir: str):
    """Create visualizations of the importance scores and metrics."""
    # Plot importance scores
    plt.figure(figsize=(12, 6))
    plt.plot(scores, 'b-', label='Importance Scores')
    plt.scatter(selected_indices, [scores[i] for i in selected_indices], 
                color='red', label='Selected Frames')
    plt.title('Frame Importance Scores')
    plt.xlabel('Frame Index')
    plt.ylabel('Importance Score')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'importance_scores.png'))
    plt.close()
    
    # Plot score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, alpha=0.7)
    plt.title('Distribution of Importance Scores')
    plt.xlabel('Importance Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'score_distribution.png'))
    plt.close()

def train_model(args):
    """Train the video summarization model."""
    # Setup feature extractor
    feature_extractor = setup_feature_extractor()
    
    # Create datasets
    train_dataset = VideoSummarizationDataset(
        features_dir=args.train_features_dir,
        scores_file=args.train_scores_file
    )
    
    val_dataset = VideoSummarizationDataset(
        features_dir=args.val_features_dir,
        scores_file=args.val_scores_file
    ) if args.val_features_dir else None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    ) if val_dataset else None
    
    # Create model
    model = TransformerSummarizer(
        feature_dim=2048,  # ResNet50 feature dimension
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    )
    
    # Create trainer
    trainer = VideoSummarizationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train model
    history = trainer.train(
        num_epochs=args.num_epochs,
        save_best=True
    )
    
    return model, history

def generate_summary(args):
    """Generate a video summary using a trained model."""
    # Select video file
    video_path = select_video_file()
    if not video_path:
        print("No video file selected. Exiting...")
        return
    
    # Create output directories
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    dirs = create_output_directories(video_name)
    
    # Create summarizer
    summarizer = VideoSummarizer(args.checkpoint_path)
    
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Calculate minimum frames needed for 20 seconds
    min_frames = int(fps * 20)  # 20 seconds worth of frames
    
    # Generate summary
    frames, scores = summarizer.generate_summary(
        video_path=video_path,
        output_dir=dirs["summary"],
        num_frames=max(min_frames, int(total_frames * 0.1)),  # At least 20 seconds or 10% of total frames
        chunk_size=100
    )
    
    # Get indices of selected frames
    selected_indices = np.argsort(scores)[-len(frames):]
    
    # Calculate metrics
    metrics = calculate_metrics(scores, selected_indices, total_frames)
    
    # Save frames
    for i, (frame, score) in enumerate(zip(frames, scores)):
        frame_path = os.path.join(dirs["frames"], f"frame_{i:04d}_score_{score:.4f}.jpg")
        cv2.imwrite(frame_path, frame)
    
    # Generate and save visualizations
    visualize_metrics(scores, selected_indices, dirs["visualizations"])
    
    # Save metrics
    with open(os.path.join(dirs["metrics"], "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Create summary video
    summary_video_path = os.path.join(dirs["base"], "summary.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(summary_video_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()
    
    print(f"\nSummary generated successfully!")
    print(f"Output saved to: {dirs['base']}")
    print(f"Total frames in video: {total_frames}")
    print(f"Selected frames: {len(frames)}")
    print(f"Summary duration: {len(frames)/fps:.2f} seconds")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"Compression ratio: {metrics['compression_ratio']:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Video Summarization System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Training arguments
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--train_features_dir", required=True,
                            help="Directory containing training features")
    train_parser.add_argument("--train_scores_file", required=True,
                            help="Path to training scores file")
    train_parser.add_argument("--val_features_dir",
                            help="Directory containing validation features")
    train_parser.add_argument("--val_scores_file",
                            help="Path to validation scores file")
    train_parser.add_argument("--batch_size", type=int, default=32,
                            help="Batch size for training")
    train_parser.add_argument("--num_workers", type=int, default=4,
                            help="Number of data loading workers")
    train_parser.add_argument("--num_epochs", type=int, default=100,
                            help="Number of training epochs")
    train_parser.add_argument("--d_model", type=int, default=512,
                            help="Transformer model dimension")
    train_parser.add_argument("--nhead", type=int, default=8,
                            help="Number of attention heads")
    train_parser.add_argument("--num_layers", type=int, default=6,
                            help="Number of transformer layers")
    train_parser.add_argument("--dim_feedforward", type=int, default=2048,
                            help="Feedforward network dimension")
    train_parser.add_argument("--dropout", type=float, default=0.1,
                            help="Dropout rate")
    train_parser.add_argument("--checkpoint_dir", default="models",
                            help="Directory to save checkpoints")
    
    # Inference arguments
    infer_parser = subparsers.add_parser("infer", help="Generate video summary")
    infer_parser.add_argument("--checkpoint_path", required=True,
                            help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_model(args)
    elif args.command == "infer":
        generate_summary(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 