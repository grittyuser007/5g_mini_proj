import os
import cv2
import argparse
import time
import torch
import threading
from src.utils.config_loader import load_config
from src.detection.injury_detector import InjuryDetector
from src.cloud.cloud_reporter import CloudReporter

def process_image(image_path, detector, reporter):
    """
    Process a single image file for injury detection and reporting
    
    Args:
        image_path: Path to the image file
        detector: InjuryDetector instance
        reporter: CloudReporter instance
    """
    print(f"Processing image: {image_path}")
    
    # Run detection on the image
    detection_result = detector.detect_from_image(image_path)
    
    if detection_result["success"]:
        print("\nDetection Results:")
        print(f"Injury Type: {detection_result['injury_type']}")
        print(f"Confidence: {detection_result['confidence']:.2f}")
        print(f"Age: {detection_result['age']}")
        print(f"Gender: {detection_result['gender']}")
        
        # Print bounding box information if available
        if detection_result.get("bbox"):
            x1, y1, x2, y2 = detection_result["bbox"]
            print(f"Bounding Box: ({x1}, {y1}) to ({x2}, {y2})")
        
        # Upload detection result to cloud
        upload_result = reporter.upload_report(detection_result, image_path)
        
        if upload_result["success"]:
            print(f"\nReport uploaded successfully. Report ID: {upload_result['report_id']}")
        else:
            print(f"\nFailed to upload report: {upload_result.get('message', 'Unknown error')}")
    else:
        print(f"\nDetection failed: {detection_result.get('error', 'Unknown error')}")

def process_video(video_path, detector, reporter):
    """
    Process a video file or camera feed for injury detection and reporting
    
    Args:
        video_path: Path to the video file or camera index
        detector: InjuryDetector instance
        reporter: CloudReporter instance
    """
    # Open video capture
    if video_path == 'camera':
        # Use camera as video source
        print("Opening camera...")
        cap = cv2.VideoCapture(detector.config['camera']['device_id'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, detector.config['camera']['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, detector.config['camera']['height'])
    else:
        # Use video file as source
        print(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
    
    # Check if the video source was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    # Process frames
    last_report_time = 0
    report_interval = 3  # Increased from 2 to 3 seconds to reduce processing frequency
    frame_count = 0
    frame_skip = 2  # Process every 3rd frame
    
    # Thread-safe variables for detection results
    latest_result = None
    processing_active = False
    
    # Function to process frames in background thread
    def process_frame_background(frame_to_process):
        nonlocal latest_result, processing_active
        processing_active = True
        result = detector.detect_from_frame(frame_to_process)
        latest_result = result
        processing_active = False
        
        # Only report to cloud if we have a significant detection
        if result["success"] and result["confidence"] > detector.confidence_threshold and result["injury_type"] != "no_injury":
            # Save frame to temporary file for cloud upload
            temp_img_path = os.path.join(detector.save_path, f"temp_frame_{frame_count}.jpg")
            cv2.imwrite(temp_img_path, frame_to_process)
            
            # Upload to cloud
            reporter.upload_report(result, temp_img_path)
            
            # Remove temporary file after upload
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            
            print(f"Detected: {result['injury_type']} - Age: {result['age']} - Gender: {result['gender']}")
    
    print("Processing video... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break
        
        frame_count += 1
        current_time = time.time()
        
        # Display original frame regardless of processing
        display_frame = frame.copy()
        
        # Skip frames to maintain smooth video but always display
        if frame_count % frame_skip != 0:
            # Just display the frame without processing
            if latest_result and latest_result["success"]:
                # Add previous detection info to current frame for display
                info_text = f"Injury: {latest_result['injury_type']} ({latest_result['confidence']:.2f})"
                person_text = f"Age: {latest_result['age']}, Gender: {latest_result['gender']}"
                
                cv2.putText(display_frame, info_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_frame, person_text, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Draw bounding box if available
                if latest_result.get("bbox"):
                    x1, y1, x2, y2 = latest_result["bbox"]
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cv2.imshow('Injury Detection System', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User requested exit.")
                break
            continue
        
        # Process frame in background thread if it's time and not already processing
        if current_time - last_report_time >= report_interval and not processing_active:
            last_report_time = current_time
            threading.Thread(target=process_frame_background, args=(frame.copy(),)).start()
        
        # Draw results on frame if available
        if latest_result and latest_result["success"]:
            info_text = f"Injury: {latest_result['injury_type']} ({latest_result['confidence']:.2f})"
            person_text = f"Age: {latest_result['age']}, Gender: {latest_result['gender']}"
            
            cv2.putText(display_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, person_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw bounding box if available
            if latest_result.get("bbox"):
                x1, y1, x2, y2 = latest_result["bbox"]
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Always display frame, whether processed or not
        cv2.imshow('Injury Detection System', display_frame)
        
        # Check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User requested exit.")
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    Main entry point for the application
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="AI-based Injury Detection System using YOLO")
    parser.add_argument("--config", default="config/config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--image", help="Path to image file to process")
    parser.add_argument("--video", help="Path to video file to process")
    parser.add_argument("--camera", action="store_true", 
                        help="Use camera for live detection")
    parser.add_argument("--gpu", action="store_true", 
                        help="Force GPU usage regardless of config setting")
    parser.add_argument("--cpu", action="store_true", 
                        help="Force CPU usage regardless of config setting")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        print("Error loading configuration. Exiting.")
        return
    
    # Override device settings if specified in command line
    if args.gpu:
        config['models']['yolo']['device'] = 'gpu'
    elif args.cpu:
        config['models']['yolo']['device'] = 'cpu'
    
    # Check for CUDA availability if GPU is requested
    if config['models']['yolo']['device'].lower() == 'gpu':
        if torch.cuda.is_available():
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        else:
            print("Warning: GPU requested but CUDA is not available. Falling back to CPU.")
            config['models']['yolo']['device'] = 'cpu'
    
    print("Initializing AI-based Injury Detection System with YOLO...")
    
    # Initialize detector and reporter
    detector = InjuryDetector(config)
    reporter = CloudReporter(config)
    
    # Process based on input type
    if args.image:
        process_image(args.image, detector, reporter)
    elif args.video:
        process_video(args.video, detector, reporter)
    elif args.camera:
        process_video("camera", detector, reporter)
    else:
        print("No input source specified. Please provide --image, --video, or --camera.")
        print("Example usage:")
        print("  python -m src.main --camera --gpu")
        print("  python -m src.main --image data/sample_images/test_image.jpg --gpu")
        print("  python -m src.main --video data/sample_videos/test_video.mp4 --gpu")

if __name__ == "__main__":
    main()