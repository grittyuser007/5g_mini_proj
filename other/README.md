# Accident Analysis Agent

A comprehensive AI-powered agent for analyzing accident victim images and generating detailed PDF reports with injury detection, demographic analysis, and location information.

## Features

- **Multi-Image Analysis**: Processes up to 5 accident victim images simultaneously
- **Injury Detection**: Detects and classifies various injury types using YOLO-based AI
- **Demographic Analysis**: Determines age groups (0-20, 21-40, 41-60, 61+) and gender
- **Bounding Box Visualization**: Draws precise bounding boxes around detected injuries
- **Location Detection**: Automatically captures incident location information
- **PDF Report Generation**: Creates comprehensive, downloadable PDF reports
- **Visual Comparison**: Shows original and annotated images side-by-side
- **Statistical Analysis**: Provides summary statistics and insights

## Supported Injury Types

- Burns
- Cuts/Lacerations
- Bruises
- Fractures
- Sprains
- Head injuries
- Abrasions
- No injury detection

## Age Group Classifications

- **0-20 years**: Young adults and children
- **21-40 years**: Young to middle-aged adults
- **41-60 years**: Middle-aged adults
- **61+ years**: Senior adults

## Installation

1. Install the additional requirements:
```bash
pip install -r other/requirements_agent.txt
```

2. Or install individual packages:
```bash
pip install reportlab geocoder pandas numpy
```

## Usage

### Command Line Usage

#### Analyze specific images:
```bash
python other/accident_analysis_agent.py --images path/to/image1.jpg path/to/image2.jpg --output my_report.pdf
```

#### Analyze all images in a directory:
```bash
python other/accident_analysis_agent.py --directory path/to/image/directory --max-images 5
```

#### Full example with cleanup:
```bash
python other/accident_analysis_agent.py --directory data/detected_images --output accident_report.pdf --cleanup
```

### Python API Usage

```python
from other.accident_analysis_agent import AccidentAnalysisAgent

# Initialize the agent
agent = AccidentAnalysisAgent()

# Method 1: Analyze specific images
image_paths = [
    "path/to/victim1.jpg",
    "path/to/victim2.jpg",
    "path/to/victim3.jpg"
]
results = agent.analyze_accident_images(image_paths)
pdf_path = agent.generate_pdf_report("my_accident_report.pdf")

# Method 2: Process entire directory
pdf_path = agent.process_accident_scene("path/to/images/", max_images=5)

# Clean up temporary files
agent.cleanup_temp_files()
```

### Demo Scripts

#### Run demo with existing sample images:
```bash
python other/demo_agent.py --mode sample
```

#### Run demo with custom images:
```bash
python other/demo_agent.py --mode custom --images image1.jpg image2.jpg
```

#### Generate test images and run demo:
```bash
python other/demo_agent.py --mode generate
```

## Report Features

The generated PDF reports include:

### 1. Executive Summary
- Report metadata and identification
- Total number of victims analyzed
- Incident location information
- Timestamp of analysis

### 2. Location Information
- City, state, and country
- GPS coordinates (when available)
- Full address information
- Incident timestamp

### 3. Statistical Overview
- Injury type distribution
- Age group demographics
- Gender distribution
- Summary insights

### 4. Detailed Image Analysis
For each analyzed image:
- Original image display
- Annotated image with bounding boxes
- Detailed analysis table
- Individual victim summary
- Confidence scores

## Output Structure

```
other/
├── analysis_outputs/           # Generated PDF reports
│   ├── accident_analysis_report_20250608_143022.pdf
│   └── accident_scene_analysis_20250608_143155.pdf
├── temp_images/               # Temporary annotated images
│   ├── annotated_image_1.jpg
│   ├── annotated_image_2.jpg
│   └── ...
├── accident_analysis_agent.py # Main agent code
├── demo_agent.py             # Demo script
├── requirements_agent.txt    # Additional requirements
└── README.md                 # This file
```

## Configuration

The agent uses the main project configuration file (`config/config.yaml`) for:
- YOLO model settings
- Confidence thresholds
- Device preferences (CPU/GPU)
- Injury classification classes

## Location Detection

The agent automatically detects the current location using IP-based geolocation. This provides:
- City and state/province
- Country information
- GPS coordinates
- Full address (when available)

**Note**: Location detection requires an internet connection. If unavailable, location fields will show "Unknown".

## Error Handling

The agent includes robust error handling for:
- Missing or corrupted image files
- Network connectivity issues for location detection
- PDF generation failures
- Temporary file management
- Configuration loading problems

## Limitations

- **Medical Accuracy**: This is an AI-based analysis tool and should not replace professional medical assessment
- **Image Quality**: Results depend on image quality, lighting, and visibility of subjects
- **Detection Limits**: Some injuries may not be visible or detectable in photographs
- **Age/Gender Detection**: Requires clear facial features for accurate demographic analysis
- **Location Accuracy**: IP-based location may not reflect the actual incident location

## Privacy and Security

- No images or data are transmitted to external services (except for location detection)
- All processing is performed locally
- Temporary files are automatically cleaned up
- Location data is optional and can be disabled

## Technical Requirements

- Python 3.8 or higher
- All dependencies from main project requirements.txt
- Additional packages: reportlab, geocoder
- Sufficient disk space for temporary image processing
- Internet connection for location detection (optional)

## Troubleshooting

### Common Issues

1. **"No analysis results available"**
   - Ensure images exist and are readable
   - Check image file formats (jpg, jpeg, png supported)
   - Verify YOLO model is properly loaded

2. **PDF generation fails**
   - Check disk space availability
   - Ensure output directory exists and is writable
   - Verify reportlab installation

3. **Location detection fails**
   - Check internet connectivity
   - This is non-critical and won't prevent report generation

4. **Low detection confidence**
   - Ensure good image quality
   - Check lighting conditions
   - Verify subjects are clearly visible

### Performance Tips

- Use GPU acceleration for faster processing (configure in config.yaml)
- Process images in smaller batches for large datasets
- Ensure sufficient system memory for image processing
- Use SSD storage for faster file operations

## Example Output

The agent generates professional PDF reports with:
- High-quality image reproduction
- Detailed analysis tables
- Statistical summaries
- Professional formatting
- Downloadable format for sharing with emergency services or medical personnel

## Support

For issues related to the agent:
1. Check the main project documentation
2. Verify all dependencies are installed
3. Ensure configuration files are properly set up
4. Check system requirements and permissions

## License

This agent is part of the AI-Based Injury Detection System project and follows the same licensing terms.
