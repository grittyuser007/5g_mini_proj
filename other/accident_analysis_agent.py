"""
Accident Analysis Agent

This agent processes accident victim images and generates comprehensive PDF reports
with injury detection, demographic analysis, and location information.

Features:
- Processes 5 accident victim images
- Detects and draws bounding boxes around injuries
- Classifies injury types
- Analyzes age groups (0-20, 21-40, 41-60, 61+) and gender
- Captures location information
- Generates downloadable PDF reports with analysis
"""

import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import uuid
import json
import geocoder
from typing import List, Dict, Tuple, Optional
import argparse

# PDF generation imports
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, Table, TableStyle, PageBreak
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# Add the parent directory to the path to import from src
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.injury_detector import InjuryDetector
from src.utils.config_loader import load_config

class AccidentAnalysisAgent:
    """
    Agent for comprehensive accident victim analysis and reporting
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """
        Initialize the accident analysis agent
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = load_config(config_path)
        self.detector = InjuryDetector(self.config)
        self.output_dir = "other/analysis_outputs"
        self.temp_dir = "other/temp_images"
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Analysis results storage
        self.analysis_results = []
        self.location_info = self._get_location_info()
        
    def _get_location_info(self) -> Dict:
        """
        Get current location information
        
        Returns:
            Dict containing location information
        """
        try:
            # Get location using IP-based geolocation
            g = geocoder.ip('me')
            
            location_info = {
                'latitude': g.latlng[0] if g.latlng else 'Unknown',
                'longitude': g.latlng[1] if g.latlng else 'Unknown',
                'city': g.city or 'Unknown',
                'state': g.state or 'Unknown',
                'country': g.country or 'Unknown',
                'address': g.address or 'Unknown',
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"Location detected: {location_info['city']}, {location_info['state']}, {location_info['country']}")
            return location_info
            
        except Exception as e:
            print(f"Warning: Could not get location information: {e}")
            return {
                'latitude': 'Unknown',
                'longitude': 'Unknown',
                'city': 'Unknown',
                'state': 'Unknown',
                'country': 'Unknown',
                'address': 'Location detection failed',
                'timestamp': datetime.now().isoformat()
            }
    
    def analyze_accident_images(self, image_paths: List[str]) -> List[Dict]:
        """
        Analyze multiple accident images
        
        Args:
            image_paths: List of paths to accident victim images
            
        Returns:
            List of analysis results for each image
        """
        print(f"Starting analysis of {len(image_paths)} accident images...")
        
        results = []
        
        for i, image_path in enumerate(image_paths[:5]):  # Limit to 5 images
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found. Skipping.")
                continue
                
            print(f"Analyzing image {i+1}/5: {os.path.basename(image_path)}")
            
            # Analyze single image
            result = self._analyze_single_image(image_path, i+1)
            if result:
                results.append(result)
        
        self.analysis_results = results
        return results
    
    def _analyze_single_image(self, image_path: str, image_number: int) -> Optional[Dict]:
        """
        Analyze a single accident image
        
        Args:
            image_path: Path to the image
            image_number: Sequential number of the image
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Load the image
            original_image = cv2.imread(image_path)
            if original_image is None:
                print(f"Error: Could not load image {image_path}")
                return None
            
            # Run injury detection
            detection_result = self.detector.detect_from_image(image_path)
            
            if not detection_result.get('success', False):
                print(f"Detection failed for {image_path}: {detection_result.get('error', 'Unknown error')}")
                return None
            
            # Create annotated image with bounding boxes and labels
            annotated_image = self._create_annotated_image(original_image, detection_result)
            
            # Save annotated image
            annotated_path = os.path.join(self.temp_dir, f"annotated_image_{image_number}.jpg")
            cv2.imwrite(annotated_path, annotated_image)
            
            # Prepare analysis result
            analysis = {
                'image_number': image_number,
                'original_path': image_path,
                'annotated_path': annotated_path,
                'filename': os.path.basename(image_path),
                'injury_type': detection_result.get('injury_type', 'Unknown'),
                'injury_confidence': detection_result.get('confidence', 0.0),
                'age': detection_result.get('age', 'Unknown'),
                'age_group': detection_result.get('age_group', 'Unknown'),
                'gender': detection_result.get('gender', 'Unknown'),
                'gender_confidence': detection_result.get('gender_confidence', 0.0),
                'bbox': detection_result.get('bbox'),
                'timestamp': datetime.now().isoformat(),
                'analysis_summary': self._generate_image_summary(detection_result)
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing image {image_path}: {str(e)}")
            return None
    
    def _create_annotated_image(self, image: np.ndarray, detection_result: Dict) -> np.ndarray:
        """
        Create an annotated image with bounding boxes and labels
        
        Args:
            image: Original image array
            detection_result: Detection results
            
        Returns:
            Annotated image array
        """
        annotated = image.copy()
        
        # Draw bounding box if available
        if detection_result.get('bbox'):
            x1, y1, x2, y2 = detection_result['bbox']
            
            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Add injury type label
            injury_label = f"Injury: {detection_result['injury_type']} ({detection_result['confidence']:.2f})"
            cv2.putText(annotated, injury_label, (x1, y1-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Add demographic information
        age_text = f"Age: {detection_result.get('age', 'Unknown')} ({detection_result.get('age_group', 'Unknown')})"
        gender_text = f"Gender: {detection_result.get('gender', 'Unknown')} ({detection_result.get('gender_confidence', 0):.2f})"
        
        cv2.putText(annotated, age_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(annotated, gender_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        return annotated
    
    def _generate_image_summary(self, detection_result: Dict) -> str:
        """
        Generate a summary statement for an individual image analysis
        
        Args:
            detection_result: Detection results for the image
            
        Returns:
            Summary statement string
        """
        injury = detection_result.get('injury_type', 'unknown injury')
        age = detection_result.get('age', 'unknown age')
        age_group = detection_result.get('age_group', 'unknown age group')
        gender = detection_result.get('gender', 'unknown gender')
        confidence = detection_result.get('confidence', 0)
        
        # Convert age group to readable format
        age_group_readable = {
            'age_0_20': '0-20 years',
            'age_21_40': '21-40 years', 
            'age_41_60': '41-60 years',
            'age_61_plus': '61+ years'
        }.get(age_group, age_group)
        
        if injury == 'no_injury':
            summary = f"Analysis shows a {gender} individual, approximately {age} years old (age group: {age_group_readable}), with no visible injuries detected in this image."
        else:
            summary = f"Analysis shows a {gender} individual, approximately {age} years old (age group: {age_group_readable}), with a detected {injury} injury (confidence: {confidence:.1%}). Medical attention may be required."
        
        return summary
    
    def generate_pdf_report(self, output_filename: Optional[str] = None) -> str:
        """
        Generate a comprehensive PDF report with all analysis results
        
        Args:
            output_filename: Optional custom filename for the PDF
            
        Returns:
            Path to the generated PDF file
        """
        if not self.analysis_results:
            raise ValueError("No analysis results available. Run analyze_accident_images first.")
        
        # Generate filename if not provided
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"accident_analysis_report_{timestamp}.pdf"
        
        pdf_path = os.path.join(self.output_dir, output_filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=20,
            textColor=colors.darkred
        )
        
        # Title
        story.append(Paragraph("ACCIDENT VICTIM ANALYSIS REPORT", title_style))
        story.append(Spacer(1, 20))
        
        # Report metadata
        report_id = str(uuid.uuid4())[:8].upper()
        story.append(Paragraph(f"Report ID: {report_id}", styles['Normal']))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"Images Analyzed: {len(self.analysis_results)}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Location Information
        story.append(Paragraph("INCIDENT LOCATION", subtitle_style))
        location_data = [
            ['Location Component', 'Value'],
            ['City', self.location_info['city']],
            ['State/Province', self.location_info['state']],
            ['Country', self.location_info['country']],
            ['Coordinates', f"{self.location_info['latitude']}, {self.location_info['longitude']}"],
            ['Address', self.location_info['address']],
            ['Timestamp', self.location_info['timestamp']]
        ]
        
        location_table = Table(location_data)
        location_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(location_table)
        story.append(Spacer(1, 30))
        
        # Summary Statistics
        story.append(Paragraph("ANALYSIS SUMMARY", subtitle_style))
        
        # Calculate statistics
        injury_types = [r['injury_type'] for r in self.analysis_results]
        age_groups = [r['age_group'] for r in self.analysis_results]
        genders = [r['gender'] for r in self.analysis_results]
        
        # Create summary statistics
        injury_counts = pd.Series(injury_types).value_counts().to_dict()
        age_group_counts = pd.Series(age_groups).value_counts().to_dict()
        gender_counts = pd.Series(genders).value_counts().to_dict()
        
        summary_text = f"""
        Total Images Analyzed: {len(self.analysis_results)}
        
        Injury Types Detected:
        {chr(10).join([f"• {injury}: {count} cases" for injury, count in injury_counts.items()])}
        
        Age Group Distribution:
        {chr(10).join([f"• {age_group.replace('age_', '').replace('_', '-') + ' years' if age_group != 'age_61_plus' else '61+ years'}: {count} individuals" for age_group, count in age_group_counts.items()])}
        
        Gender Distribution:
        {chr(10).join([f"• {gender.title()}: {count} individuals" for gender, count in gender_counts.items()])}
        """
        
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 30))
        
        # Individual Image Analysis
        story.append(Paragraph("DETAILED IMAGE ANALYSIS", subtitle_style))
        
        for i, result in enumerate(self.analysis_results):
            story.append(PageBreak())
            
            # Image header
            story.append(Paragraph(f"IMAGE {result['image_number']}: {result['filename']}", subtitle_style))
            
            # Original and annotated images side by side
            try:
                # Add original image
                story.append(Paragraph("Original Image:", styles['Heading3']))
                original_img = ReportLabImage(result['original_path'], width=4*inch, height=3*inch)
                story.append(original_img)
                story.append(Spacer(1, 10))
                
                # Add annotated image
                story.append(Paragraph("Annotated Analysis:", styles['Heading3']))
                annotated_img = ReportLabImage(result['annotated_path'], width=4*inch, height=3*inch)
                story.append(annotated_img)
                story.append(Spacer(1, 20))
                
            except Exception as e:
                story.append(Paragraph(f"Error loading images: {str(e)}", styles['Normal']))
                story.append(Spacer(1, 10))
            
            # Analysis details table
            analysis_data = [
                ['Analysis Component', 'Result', 'Confidence'],
                ['Injury Type', result['injury_type'], f"{result['injury_confidence']:.1%}"],
                ['Age', str(result['age']), 'N/A'],
                ['Age Group', result['age_group'].replace('age_', '').replace('_', '-') + ' years' if result['age_group'] != 'age_61_plus' else '61+ years', 'N/A'],
                ['Gender', result['gender'].title(), f"{result['gender_confidence']:.1%}"],
                ['Analysis Time', result['timestamp'], 'N/A']
            ]
            
            analysis_table = Table(analysis_data)
            analysis_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(analysis_table)
            story.append(Spacer(1, 20))
            
            # Analysis summary
            story.append(Paragraph("Analysis Summary:", styles['Heading4']))
            story.append(Paragraph(result['analysis_summary'], styles['Normal']))            
            story.append(Spacer(1, 30))
        
        # Build PDF
        doc.build(story)
        
        print(f"PDF report generated successfully: {pdf_path}")
        return pdf_path
    def process_accident_scene(self, image_directory: str, max_images: int = 5) -> str:
        """
        Process all images in a directory as an accident scene
        
        Args:
            image_directory: Directory containing accident victim images
            max_images: Maximum number of images to process
            
        Returns:
            Path to the generated PDF report
        """
        # Find all image files in the directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        
        for file in os.listdir(image_directory):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(image_directory, file))
        
        if not image_paths:
            raise ValueError(f"No image files found in directory: {image_directory}")
        
        # Limit to max_images
        image_paths = image_paths[:max_images]
        
        print(f"Found {len(image_paths)} images to process from {image_directory}")
        
        # Analyze images
        self.analyze_accident_images(image_paths)
        
        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"accident_scene_analysis_{timestamp}.pdf"
        
        return self.generate_pdf_report(report_filename)
    
    def cleanup_temp_files(self):
        """
        Clean up temporary files created during analysis
        """
        try:
            for file in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print("Temporary files cleaned up successfully.")
        except Exception as e:
            print(f"Warning: Could not clean up temporary files: {e}")

def main():
    """
    Main function for command-line usage
    """
    parser = argparse.ArgumentParser(description="Accident Analysis Agent - Analyze accident victim images and generate PDF reports")
    
    parser.add_argument("--images", nargs='+', help="List of image file paths to analyze")
    parser.add_argument("--directory", help="Directory containing accident victim images")
    parser.add_argument("--output", help="Output filename for the PDF report")
    parser.add_argument("--config", default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--max-images", type=int, default=5, help="Maximum number of images to process")
    parser.add_argument("--cleanup", action="store_true", help="Clean up temporary files after processing")
    
    args = parser.parse_args()
    
    if not args.images and not args.directory:
        print("Error: Please provide either --images with file paths or --directory with image directory")
        return
    
    try:
        # Initialize the agent
        agent = AccidentAnalysisAgent(args.config)
        
        # Process images
        if args.directory:
            # Process entire directory
            pdf_path = agent.process_accident_scene(args.directory, args.max_images)
        else:
            # Process specific images
            agent.analyze_accident_images(args.images[:args.max_images])
            pdf_path = agent.generate_pdf_report(args.output)
        
        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"PDF Report Generated: {pdf_path}")
        print(f"Total Images Analyzed: {len(agent.analysis_results)}")
        print(f"Location: {agent.location_info['city']}, {agent.location_info['state']}")
        print(f"{'='*60}")
        
        # Clean up if requested
        if args.cleanup:
            agent.cleanup_temp_files()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return

if __name__ == "__main__":
    main()
