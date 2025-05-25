#!/usr/bin/env python
"""
Run script for the AI-based Injury Detection System.
This script ensures the correct import paths are used when running the application.
"""

import os
import sys


# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main function from the src module
from src.main import main

if __name__ == "__main__":
    # Run the main function
    main()
