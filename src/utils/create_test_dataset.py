"""
Script to generate a sample test dataset with ground truth labels for model evaluation.
This helps in testing the model performance on age group and gender classification.
"""

import os
import cv2
import numpy as np
import pandas as pd
import argparse
from ultralytics import YOLO
from deepface import DeepFace
import random
from src.utils.config_loader import load_config

def create_sample_dataset(input_dir, output_dir, num_samples=50, config_path="config/config.yaml"):
    """
    Create a sample test dataset with ground truth labels
    
    Args:
        input_dir: Directory containing source images
        output_dir: Directory to save test images and ground truth
        num_samples: Number of test samples to create
        config_path: Path to configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # List all image files in the input directory
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    # Shuffle and sample images
    random.shuffle(image_files)
    selected_images = image_files[:min(num_samples, len(image_files))]
    
    # Prepare ground truth dataframe
    ground_truth = []
    
    # Process each selected image
    for i, image_path in enumerate(selected_images):
        try:
            print(f"Processing image {i+1}/{len(selected_images)}: {image_path}")
            
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}. Skipping.")
                continue
            
            # Detect face for age and gender
            try:
                analysis = DeepFace.analyze(
                    img_path=image,
                    actions=('age', 'gender'),
                    enforce_detection=False,
                    silent=True,
                    detector_backend='opencv'
                )
                
                if isinstance(analysis, list) and len(analysis) > 0:
                    face_analysis = analysis[0]
                    age = face_analysis.get("age", 30)
                    gender = face_analysis.get("dominant_gender", "male")
                else:
                    # Assign random values if detection fails
                    age = random.randint(20, 60)
                    gender = random.choice(['male', 'female'])
            except:
                # Assign random values if detection fails
                age = random.randint(20, 60)
                gender = random.choice(['male', 'female'])
            
            # Determine age group
            if age <= 20:
                age_group = 'age_0_20'
            elif age <= 40:
                age_group = 'age_21_40'
            elif age <= 60:
                age_group = 'age_41_60'
            else:
                age_group = 'age_61_plus'
            
            # Assign a random injury type (for demonstration)
            injury_type = random.choice(config['models']['yolo']['injury_classes'])
            
            # Save the image with a new filename
            new_filename = f"test_{i+1:03d}.jpg"
            output_path = os.path.join(output_dir, new_filename)
            cv2.imwrite(output_path, image)
            
            # Add ground truth information
            ground_truth.append({
                'filename': new_filename,
                'injury': injury_type,
                'age_group': age_group,
                'gender': gender,
                'age': age
            })
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
    
    # Save ground truth to CSV
    ground_truth_df = pd.DataFrame(ground_truth)
    ground_truth_path = os.path.join(output_dir, "ground_truth.csv")
    ground_truth_df.to_csv(ground_truth_path, index=False)
    
    print(f"\nCreated test dataset with {len(ground_truth)} images")
    print(f"Ground truth saved to {ground_truth_path}")
    
    return ground_truth_path

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create sample test dataset')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing source images')
    parser.add_argument('--output-dir', type=str, default='data/test_dataset', help='Directory to save test dataset')
    parser.add_argument('--num-samples', type=int, default=50, help='Number of test samples to create')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Create sample dataset
    ground_truth_path = create_sample_dataset(
        args.input_dir,
        args.output_dir,
        args.num_samples,
        args.config
    )
    
    print(f"\nTo evaluate the model on this test dataset, run:")
    print(f"python -m src.utils.evaluate_model --test-dir {args.output_dir} --ground-truth {ground_truth_path}")

if __name__ == '__main__':
    main()
