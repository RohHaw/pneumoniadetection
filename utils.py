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
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []

def reset_session_state():
    st.session_state.messages = []
    st.session_state.analysis_history = []

@st.cache_resource
def load_models():
    try:
        xray_validator = ChestXrayValidator("Training/Validator/xray_validator.pth")
        classifier = PneumoniaClassifier()
        return xray_validator, classifier
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def generate_results_csv(results):
    data = {
        "Class": [results["class"]],
        "Normal Probability (%)": [results["probabilities"]["Normal"]],
        "Pneumonia Probability (%)": [results["probabilities"]["Pneumonia"]],
        "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    }
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

def limit_history(history_limit):
    if len(st.session_state.analysis_history) > history_limit:
        st.session_state.analysis_history.pop(0)