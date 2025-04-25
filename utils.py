"""
Utility functions for the Pneumonia X-Ray Classifier Streamlit application.

This module provides helper functions for managing session state, loading AI models,
generating CSV results, and limiting analysis history. It supports the main application
by handling initialisation, model loading, data export, and history management, ensuring
modularity and reusability.

Author: Rohman Hawrylak
Date: April 2025
"""

import os
import sys
import streamlit as st
from datetime import datetime
import pandas as pd
from classifier import PneumoniaClassifier
from clinical_qa import ClinicalQA
sys.path.append(os.path.abspath("Training/Validator"))
from xray_validator import ChestXrayValidator

def initialise_session_state():
    """
    Initialise Streamlit session state for messages and analysis history.

    Sets up the session state variables `messages` and `analysis_history` as empty lists
    if they do not already exist, ensuring the application has a clean state for storing
    chat messages and analysis records.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Store chat messages
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []  # Store analysis history

def reset_session_state():
    """
    Reset Streamlit session state by clearing messages and analysis history.

    Clears the `messages` and `analysis_history` lists in the session state to reset
    the application's chat and analysis records.
    """
    st.session_state.messages = []  # Clear chat messages
    st.session_state.analysis_history = []  # Clear analysis history

@st.cache_resource
def load_models():
    """
    Load the chest X-ray validator and pneumonia classifier models.

    Uses Streamlit's caching to load the ResNet-18-based `ChestXrayValidator` and
    ResNet-50-based `PneumoniaClassifier` models. Returns both models if successful,
    or None for both if an error occurs.

    Returns:
        tuple: (xray_validator, classifier)
            - xray_validator (ChestXrayValidator or None): The X-ray validation model.
            - classifier (PneumoniaClassifier or None): The pneumonia classification model.
    """
    try:
        # Load the X-ray validator model
        xray_validator = ChestXrayValidator("Training/Validator/xray_validator.pth")
        # Load the pneumonia classifier model
        classifier = PneumoniaClassifier()
        return xray_validator, classifier
    except Exception as e:
        # Display error message if model loading fails
        st.error(f"Error loading models: {str(e)}")
        return None, None

def generate_results_csv(results):
    """
    Generate a CSV string from analysis results.

    Creates a pandas DataFrame from the classification results, including the predicted
    class, probabilities, and timestamp, and converts it to a CSV string without the
    index column.

    Args:
        results (dict): Dictionary containing classification results with keys 'class',
            'probabilities', and other relevant data.

    Returns:
        str: CSV string representation of the results.
    """
    # Create dictionary with result data
    data = {
        "Class": [results["class"]],
        "Normal Probability (%)": [results["probabilities"]["Normal"]],
        "Pneumonia Probability (%)": [results["probabilities"]["Pneumonia"]],
        "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    }
    # Convert to DataFrame and generate CSV
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

def limit_history(history_limit):
    """
    Limit the analysis history to a specified number of entries.

    Removes the oldest entry from the session state's `analysis_history` list if the
    number of entries exceeds the specified limit.

    Args:
        history_limit (int): Maximum number of history entries to retain.
    """
    # Remove oldest entry if history exceeds limit
    if len(st.session_state.analysis_history) > history_limit:
        st.session_state.analysis_history.pop(0)