"""
Configuration constants for the Pneumonia X-Ray Classifier application.

This module defines global configuration settings used throughout the pneumonia
classification application, including file handling and analysis parameters. These
constants ensure consistent behaviour across the application, such as supported
image formats, file size limits, confidence thresholds for X-ray validation, and
history retention limits.

Author: Rohman Hawrylak
Date: April 2025
"""

# Define supported image file types for X-ray uploads
SUPPORTED_FILE_TYPES = ["jpg", "jpeg", "png"]

# Set maximum file size limit to 10 megabytes (in bytes)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Minimum confidence percentage required for X-ray validation
CONFIDENCE_THRESHOLD = 90  # Minimum confidence for X-ray validation

# Maximum number of analysis results to retain in history
HISTORY_LIMIT = 10