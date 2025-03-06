# utils.py
"""Utility functions for the Pneumonia X-Ray Classifier app."""

import streamlit as st
from datetime import datetime
import pandas as pd
from classifier import PneumoniaClassifier
from clinical_qa import ClinicalQA
from xray_validator import ChestXrayValidator


def initialise_session_state():
    """Initialise session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []


def reset_session_state():
    """Reset session state variables to their initial values."""
    st.session_state.messages = []
    st.session_state.analysis_history = []


@st.cache_resource
def load_models():
    """Load and cache the X-ray validator and classifier models."""
    try:
        xray_validator = ChestXrayValidator("xray_validator.pth")
        classifier = PneumoniaClassifier()
        return xray_validator, classifier
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None


def generate_results_csv(results):
    """Generate a CSV string from analysis results."""
    data = {
        "Class": [results["class"]],
        "Normal Probability (%)": [results["probabilities"]["Normal"]],
        "Pneumonia Probability (%)": [results["probabilities"]["Pneumonia"]],
        "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    }
    df = pd.DataFrame(data)
    return df.to_csv(index=False)


def limit_history(history_limit):
    """Limit the size of the analysis history."""
    if len(st.session_state.analysis_history) > history_limit:
        st.session_state.analysis_history.pop(0)
        
# Add these functions to utils.py to help with rendering HTML properly

def create_html_card(title, content, bg_color="#FFFFFF", text_color="#1F2937", border_radius="5px"):
    """Create an HTML card with proper styling and contrast."""
    return f"""
    <div style='background-color: {bg_color}; 
              padding: 1rem; 
              border-radius: {border_radius}; 
              margin-bottom: 1rem;
              box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);'>
        <h4 style='color: {text_color}; margin-top: 0;'>{title}</h4>
        <div style='color: {text_color};'>{content}</div>
    </div>
    """

def render_result_badge(result_class, probability, use_dark_mode=False):
    """Create a result badge with proper contrast for both light and dark modes."""
    if result_class == "Pneumonia":
        bg_color = "#FEE2E2" if not use_dark_mode else "#7F1D1D"
        text_color = "#DC2626" if not use_dark_mode else "#FECACA"
    else:
        bg_color = "#DCFCE7" if not use_dark_mode else "#064E3B"
        text_color = "#059669" if not use_dark_mode else "#A7F3D0"
    
    return f"""
    <div style='background-color: {bg_color}; 
              padding: 1rem; 
              border-radius: 5px; 
              margin-bottom: 1rem;'>
        <h3 style='margin: 0; color: {text_color};'>{result_class}</h3>
        <p style='margin: 0; color: {text_color};'>Probability: {probability:.1f}%</p>
    </div>
    """