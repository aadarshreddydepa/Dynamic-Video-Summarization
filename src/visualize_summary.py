import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import List, Tuple

def load_summary_data(summary_dir: str) -> Tuple[List[str], List[float], List[int]]:
    """
    Load summary data from the output directory.
    
    Args:
        summary_dir (str): Path to summary directory
        
    Returns:
        Tuple[List[str], List[float], List[int]]: Frame paths, scores, and indices
    """
    summary_dir = Path(summary_dir)
    
    # Load scores
    with open(summary_dir / 'scores.json', 'r') as f:
        data = json.load(f)
        scores = data['frame_scores']
        indices = data['frame_indices']
    
    # Get frame paths
    frame_paths = sorted([str(p) for p in summary_dir.glob('frame_*.jpg')])
    
    return frame_paths, scores, indices

def plot_importance_scores(scores: List[float], indices: List[int], total_frames: int, save_path: str):
    """
    Plot importance scores over time.
    
    Args:
        scores (List[float]): Importance scores
        indices (List[int]): Frame indices
        total_frames (int): Total number of frames in video
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(15, 5))
    
    # Create x-axis values (frame numbers)
    x = np.arange(total_frames)
    y = np.zeros(total_frames)
    
    # Set scores at selected frame indices
    y[indices] = scores
    
    # Plot
    plt.plot(x, y, 'b-', alpha=0.3, label='All frames')
    plt.plot(indices, scores, 'r.', markersize=10, label='Selected frames')
    
    plt.title('Frame Importance Scores Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Importance Score')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(save_path)
    plt.close()

def create_summary_video(frame_paths: List[str], output_path: str, fps: int = 5):
    """
    Create a video from the selected frames.
    
    Args:
        frame_paths (List[str]): Paths to selected frames
        output_path (str): Path to save the output video
        fps (int): Frames per second for output video
    """
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add frames to video
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        out.write(frame)
    
    out.release()

def main():
    # Set paths
    summary_dir = 'data/summaries/sample'
    video_path = 'data/videos/sample.mp4'
    
    # Load summary data
    frame_paths, scores, indices = load_summary_data(summary_dir)
    
    # Get total frames from video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Create visualizations directory
    vis_dir = Path(summary_dir) / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    # Plot importance scores
    plot_importance_scores(
        scores=scores,
        indices=indices,
        total_frames=total_frames,
        save_path=str(vis_dir / 'importance_scores.png')
    )
    
    # Create summary video
    create_summary_video(
        frame_paths=frame_paths,
        output_path=str(vis_dir / 'summary.mp4'),
        fps=5
    )
    
    print(f'\nVisualizations saved to: {vis_dir}')
    print(f'1. Importance scores plot: {vis_dir}/importance_scores.png')
    print(f'2. Summary video: {vis_dir}/summary.mp4')

if __name__ == '__main__':
    main() 