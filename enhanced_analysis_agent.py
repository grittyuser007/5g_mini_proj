#!/usr/bin/env python3
"""
Enhanced Accident Analysis Agent using the improved detector
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from src.utils.config_loader import load_config
from src.detection.enhanced_injury_detector import EnhancedInjuryDetector

# Import the original accident analysis agent functions
original_agent_path = os.path.join(os.path.dirname(__file__), 'other', 'accident_analysis_agent.py')
sys.path.append(os.path.dirname(original_agent_path))

def create_enhanced_agent():
    """Create an enhanced accident analysis agent"""
    
    # Load configuration
    config = load_config('config/config.yaml')
    if not config:
        print("❌ Could not load configuration")
        return None
    
    # Create enhanced detector
    try:
        enhanced_detector = EnhancedInjuryDetector(config)
        print("✅ Enhanced Injury Detector loaded successfully")
        return enhanced_detector
    except Exception as e:
        print(f"❌ Error creating enhanced detector: {e}")
        return None

def analyze_image_enhanced(detector, image_path):
    """Analyze a single image with enhanced detector"""
    try:
        print(f"🔍 Analyzing: {os.path.basename(image_path)}")
        
        result = detector.detect_from_image(image_path)
        
        if result["success"]:
            print(f"✅ Detection Results:")
            print(f"   Injury Type: {result['injury_type']}")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Age: {result.get('age', 'unknown')}")
            print(f"   Gender: {result.get('gender', 'unknown')}")
            if result.get("bbox"):
                x1, y1, x2, y2 = result["bbox"]
                print(f"   Bounding Box: ({x1}, {y1}) to ({x2}, {y2})")
            print()
            return result
        else:
            print(f"❌ Detection failed: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        return None

def main():
    """Main function for enhanced analysis"""
    parser = argparse.ArgumentParser(description='Enhanced Accident Analysis with better detector')
    parser.add_argument('--image', help='Single image to analyze')
    parser.add_argument('--directory', help='Directory of images to analyze')
    parser.add_argument('--max-images', type=int, default=5, help='Maximum images to process')
    
    args = parser.parse_args()
    
    print("🚀 ENHANCED ACCIDENT ANALYSIS AGENT")
    print("=" * 50)
    
    # Create enhanced detector
    detector = create_enhanced_agent()
    if not detector:
        return 1
    
    # Process images
    if args.image:
        # Single image
        if os.path.exists(args.image):
            analyze_image_enhanced(detector, args.image)
        else:
            print(f"❌ Image not found: {args.image}")
            return 1
            
    elif args.directory:
        # Directory of images
        if not os.path.exists(args.directory):
            print(f"❌ Directory not found: {args.directory}")
            return 1
        
        # Get image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        images = [f for f in os.listdir(args.directory) 
                 if f.lower().endswith(image_extensions)]
        
        if not images:
            print(f"❌ No images found in {args.directory}")
            return 1
        
        # Limit number of images
        images = images[:args.max_images]
        print(f"📁 Found {len(images)} images to process")
        print()
        
        results = []
        for i, image_file in enumerate(images, 1):
            image_path = os.path.join(args.directory, image_file)
            print(f"🔍 Processing {i}/{len(images)}: {image_file}")
            
            result = analyze_image_enhanced(detector, image_path)
            if result:
                results.append(result)
        
        # Summary
        print("=" * 50)
        print("📊 ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Total images processed: {len(images)}")
        print(f"Successful detections: {len(results)}")
        
        if results:
            injury_types = [r['injury_type'] for r in results]
            print(f"Injury types detected: {set(injury_types)}")
            
            avg_confidence = sum(r['confidence'] for r in results) / len(results)
            print(f"Average confidence: {avg_confidence:.2f}")
    else:
        print("❌ Please specify either --image or --directory")
        return 1
    
    print("\n✅ Enhanced analysis complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
