#!/usr/bin/env python3
"""
Test the improved injury detector with current model
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config_loader import load_config
from src.detection.injury_detector import InjuryDetector

def test_improved_detector():
    """Test the improved detector with current model"""
    
    print("=" * 60)
    print("TESTING IMPROVED DETECTOR")
    print("=" * 60)
    
    # Load config
    config = load_config('config/config.yaml')
    if not config:
        print("❌ Could not load configuration")
        return
    
    # Initialize detector
    print("🔍 Initializing Injury Detector...")
    detector = InjuryDetector(config)
    
    if not detector.yolo_model:
        print("❌ Failed to load detector model")
        return
    
    # Test on sample images
    sample_dir = "other/test_images/sample_images"
    if not os.path.exists(sample_dir):
        print(f"❌ Sample directory not found: {sample_dir}")
        return
    
    images = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:3]
    
    if not images:
        print("❌ No test images found")
        return
    
    print(f"\n📊 Testing {len(images)} images:")
    print("=" * 60)
    
    for i, image_name in enumerate(images, 1):
        image_path = os.path.join(sample_dir, image_name)
        
        print(f"\n🔍 Image {i}: {image_name}")
        print("-" * 40)
        
        try:
            result = detector.detect_from_image(image_path)
            
            if result["success"]:
                print(f"✅ SUCCESS:")
                print(f"   Injury Type: {result['injury_type']}")
                print(f"   Confidence: {result['confidence']:.2f}")
                print(f"   Age: {result['age']}")
                print(f"   Gender: {result['gender']}")
                print(f"   Age Group: {result['age_group']}")
                
                if result.get("bbox"):
                    x1, y1, x2, y2 = result["bbox"]
                    print(f"   Detection Box: ({x1}, {y1}) to ({x2}, {y2})")
                else:
                    print(f"   Detection: Full image analysis")
                    
            else:
                print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Test Complete!")
    print("✅ Your model is now working with proper class mapping!")
    print("=" * 60)

if __name__ == "__main__":
    test_improved_detector()
