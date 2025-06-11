# Dynamic Video Summarization Using Transformers

This project implements a dynamic video summarization system that generates concise summaries of videos by selecting key frames based on their importance. The system uses a transformer-based architecture to process video frames and predict their importance scores.

## Features

- Dynamic frame selection based on importance scores
- Transformer-based architecture for sequence modeling
- Support for GPU acceleration
- Configurable frame sampling rate
- Feature extraction using pre-trained ResNet50
- Evaluation metrics (F1-score, precision, recall)
- Visualization tools for selected key frames

## Project Structure

```
.
├── data/                   # Directory for storing video data and features
├── models/                 # Saved model checkpoints
├── src/                    # Source code
│   ├── data/              # Data loading and processing modules
│   ├── models/            # Model architecture definitions
│   ├── training/          # Training and evaluation scripts
│   └── utils/             # Utility functions
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your video data in the `data` directory
2. Train the model:
```bash
python src/training/train.py
```

3. Generate summaries:
```bash
python src/inference.py --video_path path/to/video.mp4 --output_dir path/to/output
```

## Configuration

The system can be configured through command-line arguments or configuration files. Key parameters include:
- Frame sampling rate
- Number of key frames to select
- Model architecture parameters
- Training hyperparameters

## License

MIT License 