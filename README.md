# AI-Based Injury Detection System using YOLO

An AI-powered system for detecting injuries in accidents and analyzing victim information including age groups, gender, and injury types using YOLO (You Only Look Once) object detection. The system can process images or video streams and report analysis to the cloud.

## Features

- **YOLO-based Injury Detection**: Uses YOLOv8 to detect injuries (burns, cuts, bruises, fractures, sprains, head injuries)
- **Person Analysis**: Identifies age, age group, and gender of accident victims using DeepFace
- **Multi-Task Classification**: Classifies injury type, age group, and gender simultaneously
- **Cloud Reporting**: Automatically uploads detection reports to cloud storage (Firebase)
- **Multiple Input Sources**: Works with image files, video files, or live camera feed
- **GPU Acceleration**: Supports GPU acceleration for faster inference
- **Visual Feedback**: Displays detection results with bounding boxes and classifications
- **Optimized Processing**: Uses threading and optimized image processing for better performance

## Project Structure

```
.
├── config/                 # Configuration files
│   ├── config.yaml         # System configuration settings
│   └── firebase_credentials.json  # Firebase cloud service credentials
├── data/                   # Data storage
│   ├── custom_dataset/     # Custom training dataset with data.yaml
│   │   ├── images/         # Training and validation images
│   │   └── data.yaml       # Dataset configuration
│   ├── test_dataset/       # Test dataset for evaluation
│   └── detected_images/    # Images with detected injuries
├── docs/                   # Documentation
│   └── dataset_preparation_guide.md  # Guide for creating custom dataset
├── models/                 # ML model storage
│   └── yolov8n.pt         # YOLO model (downloaded automatically if not present)
├── src/                    # Source code
│   ├── cloud/              # Cloud reporting functionality
│   │   └── cloud_reporter.py
│   ├── detection/          # Injury detection functionality
│   │   └── injury_detector.py
│   ├── utils/              # Utility functions
│   │   ├── config_loader.py
│   │   ├── sample_data_generator.py
│   │   ├── train_model.py  # Script for training custom model
│   │   ├── evaluate_model.py  # Script for model evaluation
│   │   ├── create_test_dataset.py  # Script for creating test dataset
│   │   └── demo.py         # Demo script for showcasing capabilities
│   └── main.py             # Main application entry point
└── requirements.txt        # Project dependencies
```

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (for GPU acceleration)
- NVIDIA CUDA Toolkit and cuDNN (for GPU acceleration)
- OpenCV
- PyTorch with CUDA support
- Ultralytics YOLO
- Firebase account (for cloud reporting)

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure Firebase credentials:
   - Replace the placeholder data in `config/firebase_credentials.json` with your actual Firebase credentials

## GPU Setup

To run the system on a GPU:

1. Ensure you have NVIDIA GPU drivers installed
2. Install CUDA Toolkit compatible with your PyTorch version:
   ```bash
   # For PyTorch 2.0.1 (recommended CUDA 11.7 or 11.8)
   # Download from: https://developer.nvidia.com/cuda-downloads
   ```

3. Install cuDNN for your CUDA version:
   ```bash
   # Download from: https://developer.nvidia.com/cudnn
   ```

4. Verify GPU is detected:
   ```bash
   # Run this command to check if PyTorch can detect your GPU
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
   ```

## Usage

### Run with Camera (with GPU)

To run the system with a live camera feed using GPU:

```bash
python -m src.main --camera --gpu
```

### Run with Image File (with GPU)

To process a single image using GPU:

```bash
python -m src.main --image data/sample_images/[image_name].jpg --gpu
```

### Run with Video File (with GPU)

To process a video file using GPU:

```bash
python -m src.main --video data/sample_videos/[video_name].mp4 --gpu
```

### Force CPU Usage

If you want to force CPU usage regardless of config settings:

```bash
python -m src.main --camera --cpu
```

### Run Demo

To showcase the injury detection with age group and gender classification:

```bash
python -m src.utils.demo --input path/to/image.jpg
```

For videos:

```bash
python -m src.utils.demo --input path/to/video.mp4
```

## Custom Dataset and Training

### Creating a Custom Dataset

1. Follow the guide in `docs/dataset_preparation_guide.md` to create a properly annotated dataset
2. Use Roboflow to annotate images with injury types, age groups, and gender
3. Export dataset in YOLOv8 format to `data/custom_dataset/`

### Training a Custom Model

Train a custom YOLOv8 model for better injury detection, age group, and gender classification:

```bash
python -m src.utils.train_model --epochs 150 --batch-size 16 --img-size 640
```

### Evaluation

Evaluate the model performance on test data:

1. Create a test dataset with ground truth labels:
   ```bash
   python -m src.utils.create_test_dataset --input-dir path/to/images --output-dir data/test_dataset --num-samples 50
   ```

2. Run evaluation:
   ```bash
   python -m src.utils.evaluate_model --test-dir data/test_dataset --ground-truth data/test_dataset/ground_truth.csv
   ```

## YOLO Model Configuration

The system now uses YOLOv8 for multiple classification tasks:

1. Injury type classification (9 classes)
2. Age group classification (4 classes)
3. Gender classification (2 classes)

These are configured in `config.yaml`:

```yaml
models:
  yolo:
    path: "models/injury_detection_model.pt"  # Your custom trained model
    confidence_threshold: 0.45
    device: "gpu"  # can be "cpu" or "gpu"
    injury_classes:
      - burn
      - cut
      - bruise
      - fracture
      - sprain
      - head_injury
      - laceration
      - abrasion
      - no_injury
    age_group_classes:
      - age_0_20
      - age_21_40
      - age_41_60
      - age_61_plus
    gender_classes:
      - male
      - female
```

## Cloud Reporting

Reports include:
- Timestamp
- Injury type and confidence score
- Detected age, age group, and gender
- Image URL (if available)
- Bounding box coordinates

Data is stored in Firebase Realtime Database and images in Firebase Storage.

## Limitations

- Age and gender detection requires visible faces in the images
- Performance may vary based on image quality and lighting conditions
- For optimal results, use a custom-trained model on domain-specific data

## Future Improvements

- Implement severity scoring for injuries
- Add support for multiple injury detection on the same person
- Integrate with emergency response systems
- Develop mobile app interface for field use
- Improve performance on edge devices