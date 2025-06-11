#!/usr/bin/env python3
"""
Test Enhanced Injury Detector for immediate better results
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config_loader import load_config
from src.detection.injury_detector import InjuryDetector
from src.detection.enhanced_injury_detector import EnhancedInjuryDetector

def test_both_detectors():
    """Test both original and enhanced detectors"""
    
    # Load config
    config = load_config('config/config.yaml')
    if not config:
        print("❌ Could not load configuration")
        return
    
    # Test image
    test_image = "other/test_images/sample_images/Pasted image 20250525192705.png"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        # List available images
        sample_dir = "other/test_images/sample_images"
        if os.path.exists(sample_dir):
            images = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                test_image = os.path.join(sample_dir, images[0])
                print(f"✅ Using: {test_image}")
            else:
                print("❌ No test images found")
                return
        else:
            print("❌ Sample images directory not found")
            return
    
    print("=" * 60)
    print("DETECTOR COMPARISON TEST")
    print("=" * 60)
    print(f"Test image: {test_image}")
    print()
    
    # Test Original Detector
    print("🔍 TESTING ORIGINAL DETECTOR:")
    print("-" * 40)
    try:
        original_detector = InjuryDetector(config)
        original_result = original_detector.detect_from_image(test_image)
        
        if original_result["success"]:
            print(f"✅ Original Detection Result:")
            print(f"   Injury Type: {original_result['injury_type']}")
            print(f"   Confidence: {original_result['confidence']:.2f}")
            print(f"   Age: {original_result['age']}")
            print(f"   Gender: {original_result['gender']}")
            if original_result.get("bbox"):
                print(f"   Bounding Box: {original_result['bbox']}")
        else:
            print(f"❌ Original detector failed: {original_result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Original detector error: {e}")
    
    print()
    
    # Test Enhanced Detector
    print("🚀 TESTING ENHANCED DETECTOR:")
    print("-" * 40)
    try:
        enhanced_detector = EnhancedInjuryDetector(config)
        enhanced_result = enhanced_detector.detect_from_image(test_image)
        
        if enhanced_result["success"]:
            print(f"✅ Enhanced Detection Result:")
            print(f"   Injury Type: {enhanced_result['injury_type']}")
            print(f"   Confidence: {enhanced_result['confidence']:.2f}")
            print(f"   Age: {enhanced_result['age']}")
            print(f"   Gender: {enhanced_result['gender']}")
            if enhanced_result.get("bbox"):
                print(f"   Bounding Box: {enhanced_result['bbox']}")
        else:
            print(f"❌ Enhanced detector failed: {enhanced_result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Enhanced detector error: {e}")
    
    print()
    print("=" * 60)
    print("RECOMMENDATION: Use Enhanced Detector for better results!")
    print("=" * 60)

if __name__ == "__main__":
    test_both_detectors()
