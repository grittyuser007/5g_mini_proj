"""
Demo script to showcase injury detection with age group and gender classification
"""

import os
import cv2
import argparse
import numpy as np
from datetime import datetime
from src.detection.injury_detector import InjuryDetector
from src.utils.config_loader import load_config
from src.cloud.cloud_reporter import CloudReporter

def process_image(image_path, config_path="config/config.yaml", upload_to_cloud=False):
    """
    Process a single image and display results
    
    Args:
        image_path: Path to input image
        config_path: Path to configuration file
        upload_to_cloud: Whether to upload results to cloud
    """
    # Load configuration
    config = load_config(config_path)
    
    # Initialize detector
    detector = InjuryDetector(config)
    
    # Initialize cloud reporter if needed
    cloud_reporter = None
    if upload_to_cloud:
        cloud_reporter = CloudReporter(config)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Detect injuries, age group, and gender
    result = detector.detect_from_image(image_path)
    
    if not result.get('success', False):
        print(f"Error: Detection failed. {result.get('error', 'Unknown error')}")
        return
    
    # Create a copy of the image for annotation
    annotated_image = image.copy()
    
    # Draw bounding box if available
    if result['bbox']:
        x1, y1, x2, y2 = result['bbox']
        # Green box for injury detection
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Add detection information
    injury_text = f"Injury: {result['injury_type']} ({result['confidence']:.2f})"
    age_text = f"Age: {result['age']} - {result['age_group']}"
    gender_text = f"Gender: {result['gender']} ({result['gender_confidence']:.2f})"
    
    # Draw info on image
    cv2.putText(annotated_image, injury_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(annotated_image, age_text, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(annotated_image, gender_text, (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display results
    cv2.imshow("Injury Detection with Age Group and Gender", annotated_image)
    print("\nDetection Results:")
    print(f"Injury Type: {result['injury_type']} (Confidence: {result['confidence']:.2f})")
    print(f"Age: {result['age']} (Group: {result['age_group']})")
    print(f"Gender: {result['gender']} (Confidence: {result['gender_confidence']:.2f})")
    
    # Upload to cloud if requested
    if upload_to_cloud and cloud_reporter:
        try:
            timestamp = datetime.now().isoformat()
            upload_result = cloud_reporter.upload_detection(
                result, 
                annotated_image, 
                f"demo_{timestamp}.jpg"
            )
            if upload_result:
                print(f"Results uploaded to cloud: {upload_result}")
            else:
                print("Failed to upload results to cloud")
        except Exception as e:
            print(f"Error uploading to cloud: {str(e)}")
    
    # Save the annotated image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/detected_images/demo_{timestamp}.jpg"
    cv2.imwrite(output_path, annotated_image)
    print(f"Annotated image saved to {output_path}")
    
    # Wait for key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path, config_path="config/config.yaml", upload_to_cloud=False):
    """
    Process a video file and display results
    
    Args:
        video_path: Path to input video
        config_path: Path to configuration file
        upload_to_cloud: Whether to upload results to cloud
    """
    # Load configuration
    config = load_config(config_path)
    
    # Initialize detector
    detector = InjuryDetector(config)
    
    # Initialize cloud reporter if needed
    cloud_reporter = None
    if upload_to_cloud:
        cloud_reporter = CloudReporter(config)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/detected_images/demo_video_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Process frames
    frame_count = 0
    detection_interval = 15  # Process every 15 frames for performance
    
    print(f"Processing video: {video_path}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Only process every detection_interval frames
        if frame_count % detection_interval == 0:
            # Save frame temporarily
            temp_frame_path = f"data/detected_images/temp_frame.jpg"
            cv2.imwrite(temp_frame_path, frame)
            
            # Detect injuries, age group, and gender
            result = detector.detect_from_frame(frame)
            
            # Create annotated frame
            annotated_frame = frame.copy()
            
            if result.get('success', False):
                # Draw bounding box if available
                if result['bbox']:
                    x1, y1, x2, y2 = result['bbox']
                    # Green box for injury detection
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add detection information
                injury_text = f"Injury: {result['injury_type']} ({result['confidence']:.2f})"
                age_text = f"Age: {result['age']} - {result['age_group']}"
                gender_text = f"Gender: {result['gender']} ({result['gender_confidence']:.2f})"
                
                # Draw info on frame
                cv2.putText(annotated_frame, injury_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(annotated_frame, age_text, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(annotated_frame, gender_text, (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Upload to cloud if requested (at reduced frequency)
                if upload_to_cloud and cloud_reporter and frame_count % (detection_interval * 5) == 0:
                    try:
                        timestamp = datetime.now().isoformat()
                        upload_result = cloud_reporter.upload_detection(
                            result, 
                            annotated_frame, 
                            f"demo_video_{timestamp}.jpg"
                        )
                        if upload_result:
                            print(f"Frame results uploaded to cloud: {upload_result}")
                    except Exception as e:
                        print(f"Error uploading to cloud: {str(e)}")
            
            # Use the annotated frame
            frame = annotated_frame
        
        # Display frame
        cv2.imshow("Injury Detection with Age Group and Gender", frame)
        
        # Write frame to output video
        out.write(frame)
        
        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Clean up temp file
    if os.path.exists("data/detected_images/temp_frame.jpg"):
        os.remove("data/detected_images/temp_frame.jpg")
    
    print(f"\nProcessed video saved to {output_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Demo for injury detection with age group and gender')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or video')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    parser.add_argument('--upload', action='store_true', help='Upload results to cloud')
    
    args = parser.parse_args()
    
    # Determine input type
    if args.input.lower().endswith(('.jpg', '.jpeg', '.png')):
        process_image(args.input, args.config, args.upload)
    elif args.input.lower().endswith(('.mp4', '.avi', '.mov')):
        process_video(args.input, args.config, args.upload)
    else:
        print(f"Error: Unsupported file format: {args.input}")
        print("Supported formats: jpg, jpeg, png, mp4, avi, mov")

if __name__ == '__main__':
    main()
