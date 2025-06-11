import os
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

from inference import VideoSummarizer

class BatchProcessor:
    """Class for efficient batch processing of multiple videos."""
    
    def __init__(self, model_path: str, device: str = 'cuda', num_workers: int = 2):
        """
        Initialize the batch processor.
        
        Args:
            model_path (str): Path to the trained model
            device (str): Device to run inference on ('cuda' or 'cpu')
            num_workers (int): Number of worker threads for parallel processing
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_workers = num_workers
        self.model_path = model_path
        self.summarizer = None  # Initialize lazily
        self.queue = Queue()
        self.results = {}
        self.lock = threading.Lock()
    
    def _initialize_summarizer(self):
        """Initialize the video summarizer if not already initialized."""
        if self.summarizer is None:
            self.summarizer = VideoSummarizer(self.model_path, device=str(self.device))
    
    def process_video(self, video_path: str, output_dir: str, num_frames: int = 10) -> Dict:
        """
        Process a single video.
        
        Args:
            video_path (str): Path to input video
            output_dir (str): Directory to save summary
            num_frames (int): Number of frames to include in summary
            
        Returns:
            Dict: Processing results
        """
        try:
            self._initialize_summarizer()
            
            # Create output directory
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate summary
            frames, scores = self.summarizer.generate_summary(
                video_path=video_path,
                output_dir=str(output_dir),
                num_frames=num_frames
            )
            
            return {
                'status': 'success',
                'video_path': video_path,
                'output_dir': str(output_dir),
                'num_frames': len(frames),
                'avg_score': float(np.mean(scores))
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'video_path': video_path,
                'error': str(e)
            }
    
    def _worker(self):
        """Worker thread for processing videos."""
        while True:
            try:
                task = self.queue.get()
                if task is None:
                    break
                
                video_path, output_dir, num_frames = task
                result = self.process_video(video_path, output_dir, num_frames)
                
                with self.lock:
                    self.results[video_path] = result
                
                self.queue.task_done()
                
            except Exception as e:
                print(f"Error in worker thread: {e}")
                self.queue.task_done()
    
    def process_batch(self,
                     video_paths: List[str],
                     output_dirs: List[str],
                     num_frames: int = 10) -> Dict[str, Dict]:
        """
        Process a batch of videos in parallel.
        
        Args:
            video_paths (List[str]): List of input video paths
            output_dirs (List[str]): List of output directories
            num_frames (int): Number of frames to include in each summary
            
        Returns:
            Dict[str, Dict]: Processing results for each video
        """
        # Clear previous results
        self.results = {}
        
        # Create worker threads
        threads = []
        for _ in range(self.num_workers):
            t = threading.Thread(target=self._worker)
            t.start()
            threads.append(t)
        
        # Add tasks to queue
        for video_path, output_dir in zip(video_paths, output_dirs):
            self.queue.put((video_path, output_dir, num_frames))
        
        # Add stop signals
        for _ in range(self.num_workers):
            self.queue.put(None)
        
        # Wait for all tasks to complete
        self.queue.join()
        
        # Wait for threads to finish
        for t in threads:
            t.join()
        
        return self.results

def main():
    # Set paths
    model_path = 'models/best_model.pt'
    video_dir = Path('data/videos')
    output_base_dir = Path('data/summaries')
    
    # Get all video files
    video_paths = list(video_dir.glob('*.mp4'))
    output_dirs = [output_base_dir / video_path.stem for video_path in video_paths]
    
    # Create batch processor
    processor = BatchProcessor(
        model_path=model_path,
        device='cuda',
        num_workers=2  # Adjust based on your GPU memory
    )
    
    # Process videos
    print(f'Processing {len(video_paths)} videos...')
    results = processor.process_batch(
        video_paths=[str(p) for p in video_paths],
        output_dirs=[str(p) for p in output_dirs],
        num_frames=10
    )
    
    # Print results
    print('\nProcessing Results:')
    print('------------------')
    for video_path, result in results.items():
        print(f'\nVideo: {video_path}')
        if result['status'] == 'success':
            print(f'  Status: Success')
            print(f'  Output: {result["output_dir"]}')
            print(f'  Frames: {result["num_frames"]}')
            print(f'  Avg Score: {result["avg_score"]:.4f}')
        else:
            print(f'  Status: Error')
            print(f'  Error: {result["error"]}')

if __name__ == '__main__':
    main() 