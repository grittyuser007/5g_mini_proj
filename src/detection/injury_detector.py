import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deepface import DeepFace
from datetime import datetime

class InjuryDetector:
    """
    Class to detect injuries in images and analyze the victim's age and gender using YOLO
    """
    def __init__(self, config):
        """
        Initialize the injury detector with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Load YOLO model
        self.model_path = config['models']['yolo']['path']
        self.device = config['models']['yolo']['device']
        self.confidence_threshold = config['models']['yolo']['confidence_threshold']
        self.injury_classes = config['models']['yolo']['injury_classes']
        
        # Initialize age group and gender classes
        self.age_group_classes = config['models']['yolo']['age_group_classes']
        self.gender_classes = config['models']['yolo']['gender_classes']
        
        # Load YOLO model
        self.yolo_model = self._load_model()
        
        # Configure detection settings
        self.save_detected_images = config['detection']['save_detected_images']
        self.save_path = config['detection']['save_path']
        
        # Ensure the save directory exists
        if self.save_detected_images:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _load_model(self):
        """
        Load YOLO model
        
        Returns:
            YOLO model or None if model loading fails
        """
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                print(f"Warning: YOLO model file {self.model_path} not found. Downloading...")
                # The ultralytics library will download the model if it doesn't exist
            
            # Set the device (CPU or GPU)
            device = 0 if self.device.lower() == "gpu" and torch.cuda.is_available() else "cpu"
            if self.device.lower() == "gpu" and device == "cpu":
                print("Warning: GPU requested but not available. Using CPU instead.")
            
            # Load the YOLO model
            model = YOLO(self.model_path)
            model.to(device)  # Move model to specified device
            
            return model
            
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return None
    
    def detect_from_image(self, image_path):
        """
        Detect injuries from an image file
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Detection results including injury type, confidence, age and gender
        """
        try:
            # Check if image file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file {image_path} not found")
            
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Process the image
            return self.detect_from_frame(image, image_path)
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def detect_from_frame(self, frame, source_path=None):
        """
        Detect injuries from a video frame
        
        Args:
            frame: OpenCV image frame
            source_path: Original source path for reference
            
        Returns:
            dict: Detection results including injury type, confidence, age and gender
        """
        try:
            if self.yolo_model is None:
                raise ValueError("YOLO model not loaded successfully")
            
            # Convert BGR to RGB (YOLO expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run YOLO detection
            results = self.yolo_model(rgb_frame, conf=self.confidence_threshold)
            
            # Process the results
            injury_detections = self._process_yolo_results(results, frame)
            
            # If no injuries detected, return a default "no_injury" result
            if not injury_detections:
                # Detect age and gender using DeepFace
                age_gender = self._detect_age_gender(frame)
                result = {
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "injury_type": "no_injury",
                    "confidence": 0.0,
                    "bbox": None,
                    "age": age_gender.get("age", "unknown"),
                    "gender": age_gender.get("gender", "unknown"),
                    "gender_confidence": age_gender.get("gender_confidence", 0),
                    "age_group": age_gender.get("age_group", "unknown")
                }
                
                return result
            
            # Return the detection with the highest confidence
            highest_conf_detection = max(injury_detections, key=lambda x: x["confidence"])
            
            # Save the annotated frame if enabled
            if self.save_detected_images and source_path:
                self._save_detected_image(frame, highest_conf_detection, injury_detections)
            
            return highest_conf_detection
            
        except Exception as e:
            print(f"Error in injury detection: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    def _process_yolo_results(self, results, frame):
        """
        Process YOLO detection results and map to injury types intelligently
        
        Args:
            results: YOLO detection results
            frame: Original frame
            
        Returns:
            list: Processed detection results
        """
        detections = []
        
        # Process each detection from YOLO results
        for result in results:
            boxes = result.boxes
            
            if boxes is None:
                continue
                
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Get confidence score
                confidence = float(box.conf[0].cpu().numpy())
                  # Get class index
                class_idx = int(box.cls[0].cpu().numpy())
                
                # Process detections from our trained model (classes: male, female, injury, no injury)
                if confidence > 0.3:  # Lower threshold for our trained model
                    
                    # Map our model's classes to meaningful results
                    if class_idx < len(self.injury_classes):
                        detected_class = self.injury_classes[class_idx]
                    else:
                        # Fallback mapping for 10-class model
                        class_mapping = {
                            7: "female",  # female class
                            8: "injury",  # injury class  
                            9: "male",    # male class
                            3: "no injury"  # No Injury class
                        }
                        detected_class = class_mapping.get(class_idx, "unknown")                    # Determine injury type and gender from detected class
                    if detected_class in ["male", "female"]:
                        gender = detected_class
                        injury_type = "no injury"  # If we only detect gender, assume no injury
                    elif detected_class == "injury":
                        gender = "unknown"  # Will be determined by DeepFace
                        injury_type = "injury"
                    elif detected_class == "no injury":
                        gender = "unknown"  # Will be determined by DeepFace
                        injury_type = "no injury"
                    else:
                        gender = "unknown"
                        injury_type = detected_class
                    
                    # Get the region of interest (ROI) for age/gender analysis
                    roi = frame[y1:y2, x1:x2]
                    
                    # Skip if ROI is empty or too small
                    if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
                        continue
                    
                    # Use DeepFace for more accurate age/gender if needed
                    age_gender = self._detect_age_gender(roi)
                    
                    # Use DeepFace gender if we don't have it from detection
                    if gender == "unknown" and age_gender.get("gender"):
                        gender = age_gender.get("gender")
                    
                    # Create a detection object
                    detection = {
                        "success": True,
                        "timestamp": datetime.now().isoformat(),
                        "injury_type": injury_type,
                        "confidence": confidence,
                        "bbox": (x1, y1, x2, y2),
                        "age": age_gender.get("age", "unknown"),
                        "gender": gender,
                        "gender_confidence": confidence if gender != "unknown" else age_gender.get("gender_confidence", 0),
                        "age_group": age_gender.get("age_group", "unknown")
                    }
                    
                    detections.append(detection)
        
        return detections
    
    def _analyze_injury_from_roi(self, roi):
        """
        Analyze ROI for potential injury types using image processing
        
        Args:
            roi: Region of interest (cropped image)
            
        Returns:
            str: Inferred injury type
        """
        try:
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            
            # Analyze color characteristics
            # Red areas might indicate cuts/blood
            red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
            red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
            red_pixels = cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2)
            
            # Dark/purple areas might indicate bruises
            dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 100]))
            dark_pixels = cv2.countNonZero(dark_mask)
            
            # Bright/white areas might indicate bandages or medical attention
            bright_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
            bright_pixels = cv2.countNonZero(bright_mask)
            
            total_pixels = roi.shape[0] * roi.shape[1]
            
            if total_pixels == 0:
                return "no_injury"
            
            # Calculate ratios
            red_ratio = red_pixels / total_pixels
            dark_ratio = dark_pixels / total_pixels  
            bright_ratio = bright_pixels / total_pixels
            
            # Basic heuristics for injury classification
            if red_ratio > 0.15:
                return "cut"
            elif dark_ratio > 0.4:
                return "bruise" 
            elif bright_ratio > 0.3:
                return "laceration"  # Might be bandaged
            elif red_ratio > 0.05 or dark_ratio > 0.2:
                return "abrasion"
            else:
                return "no_injury"
                
        except Exception as e:
            print(f"Error in injury analysis: {e}")
            return "no_injury"
    def _detect_age_gender(self, frame):
        """
        Detect age and gender from a frame using DeepFace
        
        Args:
            frame: OpenCV image frame
            
        Returns:
            dict: Age and gender information with age group classification
        """
        try:
            # Skip if the frame is too small
            if frame.size == 0 or frame.shape[0] < 20 or frame.shape[1] < 20:
                return {
                    "age": None, 
                    "gender": None, 
                    "gender_confidence": 0,
                    "age_group": None
                }
            
            # Resize frame to smaller dimensions for faster processing
            # This significantly speeds up DeepFace analysis
            resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            
            # Use DeepFace for age and gender detection
            analysis = DeepFace.analyze(
                img_path=resized_frame,
                actions=('age', 'gender'),
                enforce_detection=False,
                silent=True,
                detector_backend='opencv'  # Use faster detector backend
            )
            
            # Extract the first face analysis if available
            if isinstance(analysis, list) and len(analysis) > 0:
                face_analysis = analysis[0]
                age = face_analysis.get("age", None)
                gender = face_analysis.get("dominant_gender", None)
                gender_confidence = face_analysis.get("gender", {}).get(
                    face_analysis.get("dominant_gender", ""), 0
                )
                
                # Determine age group based on detected age
                age_group = self._classify_age_group(age)
                
                return {
                    "age": age,
                    "gender": gender,
                    "gender_confidence": gender_confidence,
                    "age_group": age_group
                }
            else:
                return {
                    "age": None, 
                    "gender": None, 
                    "gender_confidence": 0,
                    "age_group": None
                }
                
        except Exception as e:
            print(f"Error in age/gender detection: {e}")
            return {
                "age": None, 
                "gender": None, 
                "gender_confidence": 0,
                "age_group": None
            }
    
    def _classify_age_group(self, age):
        """
        Classify age into predefined age groups
        
        Args:
            age: Detected age value
            
        Returns:
            str: Age group classification
        """
        if age is None:
            return None
            
        if age <= 20:
            return self.age_group_classes[0]  # age_0_20
        elif age <= 40:
            return self.age_group_classes[1]  # age_21_40
        elif age <= 60:
            return self.age_group_classes[2]  # age_41_60
        else:
            return self.age_group_classes[3]  # age_61_plus
    
    def _save_detected_image(self, frame, primary_detection, all_detections=None):
        """
        Save a frame with detection information overlaid
        
        Args:
            frame: OpenCV image frame
            primary_detection: Primary detection result to highlight
            all_detections: All detection results (optional)
        """
        try:
            # Create a copy of the frame to avoid modifying the original
            annotated_frame = frame.copy()
            
            # Draw bounding boxes for all detections if provided
            if all_detections:
                for detection in all_detections:
                    if detection["bbox"]:
                        x1, y1, x2, y2 = detection["bbox"]
                        
                        # Use different colors for different injury types
                        if detection["injury_type"] == primary_detection["injury_type"]:
                            color = (0, 255, 0)  # Green for primary detection
                        else:
                            color = (0, 165, 255)  # Orange for other detections
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Add injury type and confidence
                        injury_text = f"{detection['injury_type']} ({detection['confidence']:.2f})"
                        cv2.putText(annotated_frame, injury_text, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
              # Draw primary detection information at the top of the frame            primary_text = f"Injury: {primary_detection['injury_type']} ({primary_detection['confidence']:.2f})"
            
            # Convert age_group to readable format
            age_group_readable = primary_detection['age_group'].replace('age_', '').replace('_', '-')
            if age_group_readable == '61-plus':
                age_group_readable = '61+'
                
            # Fix gender display (change "man" to "male")
            gender = primary_detection['gender']
            if gender == "man":
                gender = "male"
                
            person_text = f"Age Group: {age_group_readable} years, Gender: {gender}"
            
            cv2.putText(annotated_frame, primary_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(annotated_frame, person_text, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Generate a unique filename based on timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.save_path}/injury_{primary_detection['injury_type']}_{timestamp}.jpg"
            
            # Save the annotated frame
            cv2.imwrite(filename, annotated_frame)
            print(f"Saved detected injury image to {filename}")
            
        except Exception as e:
            print(f"Error saving detection image: {e}")