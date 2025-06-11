#!/usr/bin/env python3
"""
Train Injury Detection Model

This script trains a YOLOv8 model for injury detection using the accident-image dataset.
The trained model will be able to detect various injuries, including burns, cuts, bruises,
and also identify gender.
"""

import os
import sys
import argparse
import yaml
import shutil
from pathlib import Path

# Add the parent directory to the path to import utility functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utils.train_model import train_custom_model

def verify_and_fix_data_yaml(data_yaml_path):
    """
    Verify and fix paths in data.yaml
    
    Args:
        data_yaml_path: Path to the data.yaml file
    """
    # Check if file exists
    if not os.path.exists(data_yaml_path):
        print(f"Error: {data_yaml_path} not found")
        return None
    
    # Load the data.yaml
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Get the base directory of the data.yaml file
    base_dir = os.path.dirname(os.path.abspath(data_yaml_path))
    
    # Original paths
    print(f"Current paths in {data_yaml_path}:")
    print(f"  train: {data.get('train', 'Not set')}")
    print(f"  val: {data.get('val', 'Not set')}")
    print(f"  test: {data.get('test', 'Not set')}")
    
    # Check if the paths need fixing
    need_fixing = False
    paths_to_check = ['train', 'val', 'test']
    fixed_paths = {}
    
    for path_key in paths_to_check:
        if path_key not in data:
            continue
            
        path_value = data[path_key]
        full_path = os.path.join(base_dir, path_value)
        
        if path_value.startswith('../') or not os.path.exists(full_path):
            need_fixing = True
            # Try to find the correct paths
            possible_paths = [
                os.path.join(base_dir, path_key, 'images'),  
                os.path.join(base_dir, path_key),            
                path_value                                   
            ]
            
            for p in possible_paths:
                if os.path.exists(p):
                    fixed_paths[path_key] = p.replace(base_dir + os.path.sep, '') 
                    print(f"  Found {path_key} directory: {p}")
                    break
            else:
                print(f"  Warning: Could not find {path_key} directory")
                fixed_paths[path_key] = path_value  
    
    # Fix the paths if needed
    if need_fixing:
        for path_key, path_value in fixed_paths.items():
            data[path_key] = path_value
        
        print("\nUpdating paths in data.yaml:")
        print(f"  train: {data.get('train', 'Not set')}")
        print(f"  val: {data.get('val', 'Not set')}")
        print(f"  test: {data.get('test', 'Not set')}")
        
        # Save the updated data.yaml
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data, f, sort_keys=False)
        
        print(f"\nUpdated {data_yaml_path} with corrected paths")
    else:
        print("\nNo path fixes needed in data.yaml")
    
    # Check if classes are defined correctly
    if 'nc' in data and 'names' in data:
        if len(data['names']) != data['nc']:
            print(f"\nWarning: Number of classes ({data['nc']}) doesn't match names list length ({len(data['names'])})")
            data['nc'] = len(data['names'])
            
            # Save the updated data.yaml
            with open(data_yaml_path, 'w') as f:
                yaml.dump(data, f, sort_keys=False)
            
            print(f"Updated {data_yaml_path} with corrected class count")
    
    return data

def backup_yolov8_model():
    """Backup the original yolov8n.pt file if it exists"""
    original_model = 'yolov8n.pt'
    backup_model = 'yolov8n.pt.backup'
    
    if os.path.exists(original_model) and not os.path.exists(backup_model):
        shutil.copy(original_model, backup_model)
        print(f"Backed up original model to {backup_model}")
    
    # Also check in models directory
    if os.path.exists('models/yolov8n.pt') and not os.path.exists('models/yolov8n.pt.backup'):
        shutil.copy('models/yolov8n.pt', 'models/yolov8n.pt.backup')
        print(f"Backed up original model in models directory")

def main():
    """Main function to parse arguments and train the model"""
    
    parser = argparse.ArgumentParser(description='Train injury detection model using YOLOv8')
    
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training (default: 16)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size for training (default: 640)')
    parser.add_argument('--data-yaml', type=str, default='accident-image.v1i.yolov8/data.yaml',
                       help='Path to the dataset YAML file')
    parser.add_argument('--device', type=str, default='0',
                       help='Device to train on (0 for GPU, cpu for CPU)')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Starting Injury Detection Model Training")
    print("=" * 50)
    print(f"Dataset: {args.data_yaml}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}")
    print(f"Device: {args.device}")
    print("=" * 50)
    
    # Backup the original model
    backup_yolov8_model()
    
    # Verify and fix the data.yaml file
    print("\nVerifying dataset configuration...")
    data_config = verify_and_fix_data_yaml(args.data_yaml)
    if data_config is None:
        print(f"Error: Could not verify dataset at {args.data_yaml}")
        return 1
    
    print(f"Classes ({data_config['nc']}): {data_config['names']}")
    
    try:
        # Train the model using the utility function
        print("\nStarting model training...")
        train_custom_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            data_yaml=args.data_yaml
        )
        
        print("\nTraining complete!")
        print("The best model is saved at: models/injury_detection_model/weights/best.pt")
        
        return 0
        
    except Exception as e:
        print(f"Error during training: {e}")
        # Try to give helpful advice based on the error
        error_str = str(e).lower()
        if "cuda" in error_str or "gpu" in error_str:
            print("\nThis seems to be a GPU-related error. Try:")
            print("1. Use CPU instead: --device cpu")
            print("2. Reduce batch size: --batch-size 8")
        elif "out of memory" in error_str:
            print("\nThis seems to be a memory error. Try:")
            print("1. Reduce batch size: --batch-size 8")
            print("2. Reduce image size: --img-size 416")
        elif "multithreading" in error_str:
            print("\nThe 'multithreading' parameter is not valid in the current Ultralytics version.")
            print("This error has been fixed in the train_model.py file. Please try again.")
        else:
            print("\nFor troubleshooting, try:")
            print("1. Reduce batch size: --batch-size 8")
            print("2. Reduce epochs for testing: --epochs 1")
            
        return 1

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
