import numpy as np
from pathlib import Path
import json
from typing import List, Tuple
import cv2
from sklearn.metrics import precision_recall_fscore_support
from scipy.spatial.distance import cosine

class SummaryEvaluator:
    """Class for evaluating video summaries."""
    
    def __init__(self, summary_dir: str):
        """
        Initialize the evaluator.
        
        Args:
            summary_dir (str): Path to summary directory
        """
        self.summary_dir = Path(summary_dir)
        self.frame_paths, self.scores, self.indices = self._load_summary_data()
    
    def _load_summary_data(self) -> Tuple[List[str], List[float], List[int]]:
        """Load summary data from the output directory."""
        # Load scores
        with open(self.summary_dir / 'scores.json', 'r') as f:
            data = json.load(f)
            scores = data['frame_scores']
            indices = data['frame_indices']
        
        # Get frame paths
        frame_paths = sorted([str(p) for p in self.summary_dir.glob('frame_*.jpg')])
        
        return frame_paths, scores, indices
    
    def compute_temporal_consistency(self) -> float:
        """
        Compute temporal consistency score.
        Measures how well the selected frames maintain temporal order.
        
        Returns:
            float: Temporal consistency score (0-1)
        """
        if len(self.indices) < 2:
            return 1.0
        
        # Check if indices are in ascending order
        is_ordered = all(self.indices[i] <= self.indices[i+1] for i in range(len(self.indices)-1))
        return 1.0 if is_ordered else 0.0
    
    def compute_diversity_score(self) -> float:
        """
        Compute diversity score based on frame features.
        Measures how diverse the selected frames are.
        
        Returns:
            float: Diversity score (0-1)
        """
        if len(self.frame_paths) < 2:
            return 1.0
        
        # Extract features for each frame
        features = []
        for frame_path in self.frame_paths:
            frame = cv2.imread(frame_path)
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            features.append(frame.flatten())
        
        features = np.array(features)
        
        # Compute pairwise cosine similarities
        similarities = []
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                sim = 1 - cosine(features[i], features[j])
                similarities.append(sim)
        
        # Diversity score is 1 - average similarity
        return 1 - np.mean(similarities)
    
    def compute_score_distribution(self) -> dict:
        """
        Compute statistics about the importance scores.
        
        Returns:
            dict: Score distribution statistics
        """
        scores = np.array(self.scores)
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores))
        }
    
    def evaluate(self) -> dict:
        """
        Run all evaluations.
        
        Returns:
            dict: Dictionary of evaluation metrics
        """
        metrics = {
            'temporal_consistency': self.compute_temporal_consistency(),
            'diversity_score': self.compute_diversity_score(),
            'score_distribution': self.compute_score_distribution(),
            'num_frames': len(self.frame_paths),
            'frame_indices': self.indices
        }
        
        return metrics

def main():
    # Set paths
    summary_dir = 'data/summaries/sample'
    
    # Create evaluator
    evaluator = SummaryEvaluator(summary_dir)
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Print results
    print('\nEvaluation Results:')
    print('------------------')
    print(f'Number of frames: {metrics["num_frames"]}')
    print(f'Temporal consistency: {metrics["temporal_consistency"]:.4f}')
    print(f'Diversity score: {metrics["diversity_score"]:.4f}')
    print('\nScore Distribution:')
    print(f'  Mean: {metrics["score_distribution"]["mean"]:.4f}')
    print(f'  Std:  {metrics["score_distribution"]["std"]:.4f}')
    print(f'  Min:  {metrics["score_distribution"]["min"]:.4f}')
    print(f'  Max:  {metrics["score_distribution"]["max"]:.4f}')
    print(f'  Median: {metrics["score_distribution"]["median"]:.4f}')
    
    # Save metrics
    metrics_path = Path(summary_dir) / 'evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f'\nMetrics saved to: {metrics_path}')

if __name__ == '__main__':
    main() 