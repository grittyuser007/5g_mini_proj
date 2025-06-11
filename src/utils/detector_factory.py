#!/usr/bin/env python3
"""
Smart Detector Factory - Automatically chooses the best detector
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.detection.injury_detector import InjuryDetector
from src.detection.enhanced_injury_detector import EnhancedInjuryDetector

def create_injury_detector(config):
    """
    Smart factory to create the best available injury detector
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Best available detector instance
    """
    
    # Check if enhanced mode is enabled in config
    enhanced_mode = config.get('models', {}).get('yolo', {}).get('enhanced_mode', False)
    
    if enhanced_mode:
        try:
            print("🚀 Creating Enhanced Injury Detector...")
            detector = EnhancedInjuryDetector(config)
            
            # Verify both models loaded
            if hasattr(detector, 'trained_model') and hasattr(detector, 'base_model'):
                if detector.trained_model or detector.base_model:
                    print("✅ Enhanced Detector initialized successfully!")
                    return detector
                else:
                    print("⚠️ Enhanced Detector failed to load models, falling back to standard detector")
            else:
                print("⚠️ Enhanced Detector not properly initialized, falling back to standard detector")
                
        except Exception as e:
            print(f"⚠️ Enhanced Detector failed: {e}")
            print("🔄 Falling back to standard detector...")
    
    # Fallback to standard detector
    print("🔍 Creating Standard Injury Detector...")
    try:
        detector = InjuryDetector(config)
        print("✅ Standard Detector initialized successfully!")
        return detector
    except Exception as e:
        print(f"❌ Failed to create any detector: {e}")
        return None

def test_detector_factory():
    """Test the detector factory"""
    from src.utils.config_loader import load_config
    
    config = load_config('config/config.yaml')
    if not config:
        print("❌ Could not load configuration")
        return
    
    print("=" * 50)
    print("SMART DETECTOR FACTORY TEST")
    print("=" * 50)
    
    # Test with enhanced mode
    print("1. Testing with enhanced_mode = True")
    config['models']['yolo']['enhanced_mode'] = True
    detector1 = create_injury_detector(config)
    print(f"   Result: {type(detector1).__name__}")
    print()
    
    # Test with enhanced mode disabled
    print("2. Testing with enhanced_mode = False")
    config['models']['yolo']['enhanced_mode'] = False
    detector2 = create_injury_detector(config)
    print(f"   Result: {type(detector2).__name__}")
    print()
    
    # Test a quick detection
    if detector1:
        test_image = "other/test_images/sample_images"
        if os.path.exists(test_image):
            images = [f for f in os.listdir(test_image) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                test_path = os.path.join(test_image, images[0])
                print(f"3. Quick detection test with {images[0]}")
                try:
                    result = detector1.detect_from_image(test_path)
                    if result["success"]:
                        print(f"   ✅ {result['injury_type']} (confidence: {result['confidence']:.2f})")
                    else:
                        print(f"   ❌ Detection failed")
                except Exception as e:
                    print(f"   ❌ Error: {e}")
    
    print("=" * 50)
    print("✅ Detector Factory Test Complete!")

if __name__ == "__main__":
    test_detector_factory()
