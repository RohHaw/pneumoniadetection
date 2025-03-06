# config.py
"""Configuration constants for the Pneumonia X-Ray Classifier app."""

SUPPORTED_FILE_TYPES = ["jpg", "jpeg", "png"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
CONFIDENCE_THRESHOLD = 90  # Minimum confidence for X-ray validation
HISTORY_LIMIT = 10  # Max number of analyses to store in history