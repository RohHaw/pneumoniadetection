import streamlit as st
import numpy as np
from PIL import Image
from classifier import PneumoniaClassifier
from clinical_qa import ClinicalQA
import os
import io
import pandas as pd
from datetime import datetime

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

def modern_result_display(results, original_image, gradcam_image):
    """Create a modern, card-based result display"""
    # Results cards
    col1, col2 = st.columns([1, 1])
    
    # Classification Result
    with col1:
        classification_color = "red" if results["class"] == "Pneumonia" else "green"
        st.markdown(f"""
        <div style='background-color: #f0f2f6; border-radius: 10px; padding: 20px; margin-bottom: 20px;'>
            <h3 style='color: #333;'>Classification Result</h3>
            <div style='text-align: center;'>
                <h2 style='color: {classification_color};'>{results["class"]}</h2>
                <p style='font-size: 1.2em;'>Confidence: {results["confidence"]:.1f}%</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Probability Distribution
    with col2:
        st.markdown(f"""
        <div style='background-color: #f0f2f6; border-radius: 10px; padding: 20px; margin-bottom: 20px;'>
            <h3 style='color: #333;'>Probability Distribution</h3>
            <div style='text-align: center;'>
                <div style='margin: 10px 0;'>
                    <div>Normal: {results["probabilities"]["Normal"]:.1f}%</div>
                    <div style='background-color: #e0e0e0; border-radius: 5px; margin: 5px 0;'>
                        <div style='background-color: #4CAF50; width: {results["probabilities"]["Normal"]}%; height: 20px; border-radius: 5px;'></div>
                    </div>
                </div>
                <div style='margin: 10px 0;'>
                    <div>Pneumonia: {results["probabilities"]["Pneumonia"]:.1f}%</div>
                    <div style='background-color: #e0e0e0; border-radius: 5px; margin: 5px 0;'>
                        <div style='background-color: #f44336; width: {results["probabilities"]["Pneumonia"]}%; height: 20px; border-radius: 5px;'></div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Image Analysis Section
    st.markdown("### X-Ray Analysis")
    
    image_col1, image_col2 = st.columns(2)
    
    with image_col1:
        st.image(original_image, caption="Original X-Ray", use_container_width=True)
        
    with image_col2:
        st.image(gradcam_image, caption="Areas of Interest (GradCAM)", use_container_width=True)

def main():
    st.set_page_config(page_title="Pneumonia X-Ray Classifier", layout="wide", page_icon="🩺")
    initialize_session_state()

    st.title("🩺 Pneumonia X-Ray Classification")
    st.markdown("""
    Upload a chest X-ray image for automated pneumonia detection. 
    The system will analyze the image and provide detailed results with visual explanations.
    """)

    # Get API key
    api_key = os.getenv('GEMINI_API_KEY') or st.secrets['GEMINI_API_KEY']
    
    # Initialize systems
    try:
        classifier = PneumoniaClassifier()
        qa_system = ClinicalQA(api_key=api_key)
        model_load_success = True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        model_load_success = False

    if model_load_success:
        uploaded_file = st.file_uploader(
            "Upload Chest X-Ray",
            type=["jpg", "jpeg", "png"],
            help="Upload a chest X-ray image for analysis"
        )

        if uploaded_file is not None:
            # Process and display results
            image = Image.open(uploaded_file).convert('RGB')
        
        
            
            with st.spinner('Analyzing image...'):
                try:
                    results = classifier.predict(image)
                    qa_system.set_context(results, image)
                    modern_result_display(results, image, results['gradcam'])
                    
                    # Add to analysis history
                    st.session_state.analysis_history.append({
                        'timestamp': datetime.now(),
                        'results': results,
                        'image': image
                    })
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    return

            # Chat interface with common questions
            st.markdown("---")
            st.subheader("🤖 Clinical Assistant")

            # Chat input
            if prompt := st.chat_input("Ask a custom question about the X-ray analysis"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Generating response..."):
                        response = qa_system.answer_question(prompt)
                        st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})

        else:
            st.info("👆 Upload a chest X-ray image to begin analysis")

if __name__ == "__main__":
    main()
