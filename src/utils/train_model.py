import os
import argparse
from ultralytics import YOLO

def train_custom_model(epochs=100, batch_size=16, img_size=640, data_yaml='data/custom_dataset/data.yaml'):
    """
    Train a custom YOLO model for injury detection with age group and gender classification
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Image size for training
        data_yaml: Path to the dataset YAML file
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Initialize the model - use YOLOv8n as base model
    model = YOLO('yolov8n.pt')  # Load a pretrained YOLOv8n model
    
    # Set training arguments
    args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'patience': 50,  # Early stopping patience
        'device': 0 if model.device.type == 'cuda' else 'cpu',
        'project': 'models',
        'name': 'injury_detection_model',
        'exist_ok': True,
        'task': 'detect',         # Detection task
        'single_cls': False,      # Multiple classes
        'nms': True,              # Apply NMS
        'iou': 0.7,               # IOU threshold for NMS
        'conf': 0.001,            # Low confidence threshold for training
        'rect': True,             # Use rectangular training
        'cos_lr': True,           # Use cosine learning rate
        'multithreading': True    # Enable multithreading for faster training
    }
    
    # Start training
    print(f"Starting training with {epochs} epochs, batch size {batch_size}, image size {img_size}")
    print(f"Training will include injury, age group, and gender classifications")
    results = model.train(**args)
    
    # Use the best model weights
    best_model_path = os.path.join('models', 'injury_detection_model', 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        # Copy the best model to the models directory
        import shutil
        shutil.copy(best_model_path, 'models/injury_detection_model.pt')
        print(f"Best model saved to models/injury_detection_model.pt")
    else:
        print(f"Warning: Best model not found at {best_model_path}")
        # Use the last model
        last_model_path = os.path.join('models', 'injury_detection_model', 'weights', 'last.pt')
        if os.path.exists(last_model_path):
            shutil.copy(last_model_path, 'models/injury_detection_model.pt')
            print(f"Last model saved to models/injury_detection_model.pt")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Train a custom YOLO model for injury detection')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for training')
    parser.add_argument('--data', type=str, default='data/custom_dataset/data.yaml', help='Path to the dataset YAML file')
    
    args = parser.parse_args()
    
    train_custom_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        data_yaml=args.data
    )

if __name__ == '__main__':
    main()