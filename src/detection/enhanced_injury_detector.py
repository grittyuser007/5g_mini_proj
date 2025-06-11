#!/usr/bin/env python3
"""
Enhanced Injury Detector with Hybrid Approach
Combines custom trained model with base YOLO for better performance
"""

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deepface import DeepFace
from datetime import datetime

class EnhancedInjuryDetector:
    """
    Enhanced injury detector that uses both custom trained model and base YOLO
    """
    def __init__(self, config):
        self.config = config
        
        # Load both models
        self.trained_model_path = config['models']['yolo']['path']
        self.base_model_path = 'yolov8n.pt'
        
        self.device = config['models']['yolo']['device']
        self.confidence_threshold = config['models']['yolo']['confidence_threshold']
        self.injury_classes = config['models']['yolo']['injury_classes']
        
        # Initialize age group and gender classes
        self.age_group_classes = config['models']['yolo']['age_group_classes']
        self.gender_classes = config['models']['yolo']['gender_classes']
        
        # Load both models
        self.trained_model = self._load_model(self.trained_model_path, "trained")
        self.base_model = self._load_model(self.base_model_path, "base")
        
        # Configure detection settings
        self.save_detected_images = config['detection']['save_detected_images']
        self.save_path = config['detection']['save_path']
        
        if self.save_detected_images:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _load_model(self, model_path, model_type):
        """Load a YOLO model with error handling"""
        try:
            if not os.path.exists(model_path):
                print(f"Warning: {model_type} model file {model_path} not found")
                return None
              # Load model with proper error handling for PyTorch compatibility
            try:
                # Try new PyTorch method first
                torch.serialization.add_safe_globals([
                    'ultralytics.nn.tasks.DetectionModel',
                    'ultralytics.models.yolo.detect.DetectionPredictor',
                    'ultralytics.models.yolo.detect.DetectionValidator'
                ])
            except AttributeError:
                # For older PyTorch versions, this is not needed
                pass
            
            # Load with weights_only=False for compatibility
            model = YOLO(model_path)
            
            # Set device
            device = 0 if self.device.lower() == "gpu" and torch.cuda.is_available() else "cpu"
            model.to(device)
            
            print(f"✅ {model_type.capitalize()} model loaded: {model_path}")
            print(f"   Classes: {len(model.names)} | Device: {device}")
            
            return model
            
        except Exception as e:
            print(f"❌ Error loading {model_type} model: {e}")
            return None
    
    def detect_from_image(self, image_path):
        """Enhanced detection using hybrid approach"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file {image_path} not found")
            
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            return self.detect_from_frame(image, image_path)
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return {"success": False, "error": str(e)}
    
    def detect_from_frame(self, frame, source_path=None):
        """Enhanced detection with hybrid approach"""
        try:
            # First try trained model
            if self.trained_model:
                trained_result = self._detect_with_model(frame, self.trained_model, "trained")
                if trained_result and trained_result.get("confidence", 0) > 0.3:
                    print(f"✅ Using trained model result (confidence: {trained_result['confidence']:.2f})")
                    return trained_result
            
            # Fallback to base model for person detection
            if self.base_model:
                base_result = self._detect_with_base_model(frame)
                if base_result:
                    print(f"⚠️ Using base model fallback")
                    return base_result
            
            # No detection - return no injury result
            age_gender = self._detect_age_gender(frame)
            return {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "injury_type": "No Injury",
                "confidence": 0.8,  # High confidence for no injury
                "bbox": None,
                "age": age_gender.get("age", "unknown"),
                "gender": age_gender.get("gender", "unknown"),
                "gender_confidence": age_gender.get("gender_confidence", 0),
                "age_group": age_gender.get("age_group", "unknown")
            }
            
        except Exception as e:
            print(f"Error in enhanced detection: {e}")
            return {"success": False, "error": str(e)}
    
    def _detect_with_model(self, frame, model, model_type):
        """Detect using specified model"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(rgb_frame, conf=self.confidence_threshold)
            
            best_detection = None
            best_confidence = 0
            
            for result in results:
                if not hasattr(result, 'boxes') or result.boxes is None:
                    continue
                    
                for box in result.boxes:
                    confidence = float(box.conf[0].cpu().numpy())
                    class_idx = int(box.cls[0].cpu().numpy())
                    
                    if confidence > best_confidence:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        # Map class to injury type
                        if model_type == "trained" and class_idx < len(self.injury_classes):
                            injury_type = self.injury_classes[class_idx]
                        else:
                            # For base model, map to general injury types
                            injury_type = "injury"  # Generic injury detection
                        
                        # Get age/gender from the detected region
                        roi = frame[y1:y2, x1:x2]
                        age_gender = self._detect_age_gender(roi)
                        
                        best_detection = {
                            "success": True,
                            "timestamp": datetime.now().isoformat(),
                            "injury_type": injury_type,
                            "confidence": confidence,
                            "bbox": (x1, y1, x2, y2),
                            "age": age_gender.get("age", "unknown"),
                            "gender": age_gender.get("gender", "unknown"),
                            "gender_confidence": age_gender.get("gender_confidence", 0),
                            "age_group": age_gender.get("age_group", "unknown")
                        }
                        best_confidence = confidence
            
            return best_detection
            
        except Exception as e:
            print(f"Error in {model_type} model detection: {e}")
            return None
    
    def _detect_with_base_model(self, frame):
        """Use base model to detect people and infer injury"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.base_model(rgb_frame, conf=0.25)  # Lower threshold for person detection
            
            for result in results:
                if not hasattr(result, 'boxes') or result.boxes is None:
                    continue
                    
                for box in result.boxes:
                    class_idx = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Check if it's a person (class 0 in COCO)
                    if class_idx == 0 and confidence > 0.5:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        # Analyze the person for potential injuries
                        roi = frame[y1:y2, x1:x2]
                        age_gender = self._detect_age_gender(roi)
                        
                        # Use heuristics to determine injury type
                        injury_type = self._infer_injury_type(roi)
                        
                        return {
                            "success": True,
                            "timestamp": datetime.now().isoformat(),
                            "injury_type": injury_type,
                            "confidence": confidence * 0.7,  # Reduce confidence for inference
                            "bbox": (x1, y1, x2, y2),
                            "age": age_gender.get("age", "unknown"),
                            "gender": age_gender.get("gender", "unknown"),
                            "gender_confidence": age_gender.get("gender_confidence", 0),
                            "age_group": age_gender.get("age_group", "unknown")
                        }
            
            return None
            
        except Exception as e:
            print(f"Error in base model detection: {e}")
            return None
    
    def _infer_injury_type(self, roi):
        """Simple heuristic to infer injury type from image characteristics"""
        try:
            # Analyze color distribution for potential injuries
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Check for red areas (potential cuts/blood)
            red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
            red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
            red_pixels = cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2)
            
            # Check for dark areas (potential bruises)
            dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
            dark_pixels = cv2.countNonZero(dark_mask)
            
            total_pixels = roi.shape[0] * roi.shape[1]
            
            if total_pixels == 0:
                return "No Injury"
            
            red_ratio = red_pixels / total_pixels
            dark_ratio = dark_pixels / total_pixels
            
            if red_ratio > 0.1:
                return "cut"
            elif dark_ratio > 0.3:
                return "bruise"
            else:
                return "injury"  # Generic injury
                
        except Exception:
            return "injury"  # Default to generic injury
    
    def _detect_age_gender(self, frame):
        """Detect age and gender using DeepFace"""
        try:
            if frame.size == 0 or frame.shape[0] < 20 or frame.shape[1] < 20:
                return {"age": None, "gender": None, "gender_confidence": 0, "age_group": None}
            
            # Resize for faster processing
            resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            
            analysis = DeepFace.analyze(
                img_path=resized_frame,
                actions=('age', 'gender'),
                enforce_detection=False,
                silent=True,
                detector_backend='opencv'
            )
            
            if isinstance(analysis, list) and len(analysis) > 0:
                face_analysis = analysis[0]
                age = face_analysis.get("age", None)
                gender = face_analysis.get("dominant_gender", None)
                gender_confidence = face_analysis.get("gender", {}).get(
                    face_analysis.get("dominant_gender", ""), 0
                )
                
                age_group = self._classify_age_group(age)
                
                return {
                    "age": age,
                    "gender": gender,
                    "gender_confidence": gender_confidence,
                    "age_group": age_group
                }
            else:
                return {"age": None, "gender": None, "gender_confidence": 0, "age_group": None}
                
        except Exception as e:
            print(f"Error in age/gender detection: {e}")
            return {"age": None, "gender": None, "gender_confidence": 0, "age_group": None}
    
    def _classify_age_group(self, age):
        """Classify age into groups"""
        if age is None:
            return None
            
        if age <= 20:
            return self.age_group_classes[0]
        elif age <= 40:
            return self.age_group_classes[1]
        elif age <= 60:
            return self.age_group_classes[2]
        else:
            return self.age_group_classes[3]
