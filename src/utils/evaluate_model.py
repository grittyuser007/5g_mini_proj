"""
Script to evaluate the performance of the custom YOLO model for injury detection,
age group classification, and gender recognition.
"""

import os
import cv2
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix, classification_report
from src.detection.injury_detector import InjuryDetector
from src.utils.config_loader import load_config

def evaluate_model(test_dir, ground_truth_file, config_path="config/config.yaml"):
    """
    Evaluate the model performance on test images with ground truth labels
    
    Args:
        test_dir: Directory containing test images
        ground_truth_file: CSV file with ground truth labels (format: filename,injury,age_group,gender)
        config_path: Path to configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Initialize detector
    detector = InjuryDetector(config)
    
    # Load ground truth data
    ground_truth = {}
    with open(ground_truth_file, 'r') as f:
        # Skip header
        next(f)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                filename = parts[0]
                ground_truth[filename] = {
                    'injury': parts[1],
                    'age_group': parts[2],
                    'gender': parts[3]
                }
    
    # Prepare results containers
    injury_true = []
    injury_pred = []
    age_group_true = []
    age_group_pred = []
    gender_true = []
    gender_pred = []
    
    # Process each test image
    print(f"Evaluating model on {len(ground_truth)} test images...")
    
    for filename, labels in ground_truth.items():
        image_path = os.path.join(test_dir, filename)
        
        # Skip if file doesn't exist
        if not os.path.exists(image_path):
            print(f"Warning: File {image_path} not found. Skipping.")
            continue
        
        # Detect injuries, age group, and gender
        result = detector.detect_from_image(image_path)
        
        if result.get('success', False):
            # Append true and predicted values for evaluation
            injury_true.append(labels['injury'])
            injury_pred.append(result['injury_type'])
            
            age_group_true.append(labels['age_group'])
            age_group_pred.append(result['age_group'])
            
            gender_true.append(labels['gender'])
            gender_pred.append(result['gender'])
        else:
            print(f"Warning: Detection failed for {image_path}. Error: {result.get('error', 'Unknown error')}")
    
    # Calculate and print evaluation metrics
    print("\n===== INJURY CLASSIFICATION REPORT =====")
    print(classification_report(injury_true, injury_pred))
    
    print("\n===== AGE GROUP CLASSIFICATION REPORT =====")
    print(classification_report(age_group_true, age_group_pred))
    
    print("\n===== GENDER CLASSIFICATION REPORT =====")
    print(classification_report(gender_true, gender_pred))
    
    # Calculate confusion matrices
    injury_cm = confusion_matrix(injury_true, injury_pred, labels=config['models']['yolo']['injury_classes'])
    age_group_cm = confusion_matrix(age_group_true, age_group_pred, labels=config['models']['yolo']['age_group_classes'])
    gender_cm = confusion_matrix(gender_true, gender_pred, labels=config['models']['yolo']['gender_classes'])
    
    # Print confusion matrices
    print("\n===== INJURY CONFUSION MATRIX =====")
    print_confusion_matrix(injury_cm, config['models']['yolo']['injury_classes'])
    
    print("\n===== AGE GROUP CONFUSION MATRIX =====")
    print_confusion_matrix(age_group_cm, config['models']['yolo']['age_group_classes'])
    
    print("\n===== GENDER CONFUSION MATRIX =====")
    print_confusion_matrix(gender_cm, config['models']['yolo']['gender_classes'])
    
    return {
        'injury_accuracy': np.mean(np.array(injury_true) == np.array(injury_pred)),
        'age_group_accuracy': np.mean(np.array(age_group_true) == np.array(age_group_pred)),
        'gender_accuracy': np.mean(np.array(gender_true) == np.array(gender_pred))
    }

def print_confusion_matrix(cm, class_names):
    """
    Print a confusion matrix in a readable format
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
    """
    # Print header
    header = "True\\Pred"
    for name in class_names:
        header += f"\t{name[:7]}"
    print(header)
    
    # Print rows
    for i, name in enumerate(class_names):
        row = f"{name[:7]}"
        for j in range(len(class_names)):
            row += f"\t{cm[i, j]}"
        print(row)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate custom YOLO model')
    parser.add_argument('--test-dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--ground-truth', type=str, required=True, help='CSV file with ground truth labels')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_model(args.test_dir, args.ground_truth, args.config)
    
    # Print overall accuracy
    print("\n===== OVERALL ACCURACY =====")
    print(f"Injury Classification Accuracy: {results['injury_accuracy']:.4f}")
    print(f"Age Group Classification Accuracy: {results['age_group_accuracy']:.4f}")
    print(f"Gender Classification Accuracy: {results['gender_accuracy']:.4f}")
    print(f"Combined Accuracy: {(results['injury_accuracy'] + results['age_group_accuracy'] + results['gender_accuracy'])/3:.4f}")

if __name__ == '__main__':
    main()
