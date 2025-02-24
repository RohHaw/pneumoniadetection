import streamlit as st
import numpy as np
from PIL import Image
from classifier import PneumoniaClassifier
from clinical_qa import ClinicalQA
import os
import io
import pandas as pd
from datetime import datetime
from xray_validator import ChestXrayValidator

def initialise_session_state():
    """Initialise session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

def modern_result_display(results, original_image, gradcam_image):
    """Create a modern, card-based result display"""
    # Results cards
    col1, col2 = st.columns([1, 1])
    
    # Calculate normalised confidence intervals for display
    normal_ci = results["confidence_interval"]["Normal"]
    pneumonia_ci = results["confidence_interval"]["Pneumonia"]

    st.markdown(f"""
    <div style='background-color: #f0f2f6; border-radius: 10px; padding: 20px; margin-bottom: 20px;'>
        <h3 style='color: #333;'>Probability Distribution</h3>
        <div style='text-align: center;'>
            <div style='margin: 10px 0;'>
                <div style='color: black;'>Normal: {results["probabilities"]["Normal"]:.1f}% 
                     <span style='font-size: 0.9em; color: black;'>(95% CI: {normal_ci[0]:.1f}% - {normal_ci[1]:.1f}%)</span>
                </div>
                <div style='background-color: #e0e0e0; border-radius: 5px; margin: 5px 0; position: relative;'>
                    <div style='background-color: #4CAF50; width: {results["probabilities"]["Normal"]}%; height: 20px; border-radius: 5px;'></div>
                    <div style='position: absolute; top: 0; left: {max(0, normal_ci[0])}%; width: {min(100, normal_ci[1]) - max(0, normal_ci[0])}%; height: 20px; background-color: rgba(76, 175, 80, 0.3); border-radius: 5px;'></div>
                </div>
            </div>
            <div style='margin: 10px 0;'>
                <div style='color: black;'>Pneumonia: {results["probabilities"]["Pneumonia"]:.1f}%
                     <span style='font-size: 0.9em; color: black;'>(95% CI: {pneumonia_ci[0]:.1f}% - {pneumonia_ci[1]:.1f}%)</span>
                </div>
                <div style='background-color: #e0e0e0; border-radius: 5px; margin: 5px 0; position: relative;'>
                    <div style='background-color: #f44336; width: {results["probabilities"]["Pneumonia"]}%; height: 20px; border-radius: 5px;'></div>
                    <div style='position: absolute; top: 0; left: {max(0, pneumonia_ci[0])}%; width: {min(100, pneumonia_ci[1]) - max(0, pneumonia_ci[0])}%; height: 20px; background-color: rgba(244, 67, 54, 0.3); border-radius: 5px;'></div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Image Analysis Section
    st.markdown("### X-Ray Analysis")
    
    # Determine if we should show GradCAM based on prediction and confidence
    show_gradcam = (
        results["class"] == "Pneumonia" or  # Always show for pneumonia cases
        results["probabilities"]["Pneumonia"] > 20 or  # Show if there's significant pneumonia probability
        results["uncertainty"] > 10  # Show if there's high uncertainty
    )
    
    if show_gradcam:
        image_col1, image_col2 = st.columns(2)
        
        with image_col1:
            st.image(original_image, caption="Original X-Ray", use_container_width=True)
            
        with image_col2:
            st.image(gradcam_image, caption="Areas of Interest (GradCAM)", use_container_width=True)
            if results["class"] == "Normal":
                st.info("GradCAM visualisation is shown due to either elevated pneumonia probability or model uncertainty.")
    else:
        # Show only the original image if GradCAM isn't relevant
        st.image(original_image, caption="Original X-Ray", use_container_width=True)
        st.info("GradCAM visualisation is not shown for normal X-rays with high confidence.")

def main():
    st.set_page_config(page_title="Pneumonia X-Ray Classifier", layout="wide", page_icon="🩺")
    initialise_session_state()

    st.title("🩺 Pneumonia X-Ray Classification")
    st.markdown("""
    Upload a chest X-ray image for automated pneumonia detection. 
    The system will first validate that the uploaded image is a chest X-ray before proceeding with the analysis.
    """)

    # Get API key
    api_key = os.getenv('GEMINI_API_KEY') or st.secrets['GEMINI_API_KEY']

    # Initialise systems
    try:
        xray_validator = ChestXrayValidator("xray_validator.pth")
        classifier = PneumoniaClassifier()
        qa_system = ClinicalQA(api_key=api_key)
        model_load_success = True
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        model_load_success = False

    if model_load_success:
        uploaded_file = st.file_uploader(
            "Upload Chest X-Ray",
            type=["jpg", "jpeg", "png"],
            help="Upload a chest X-ray image for analysis"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            
            # First validate if it's a chest X-ray
            with st.spinner('Validating image...'):
                is_xray, confidence = xray_validator.validate_image(image)
                
                if not is_xray:
                    st.error(f"The uploaded image does not appear to be a chest X-ray (confidence: {confidence:.1f}%). Please upload a valid chest X-ray image.")
                    return
                
                if confidence < 90:
                    st.warning(f"The system is not completely confident this is a chest X-ray (confidence: {confidence:.1f}%). Please verify the image is correct.")

            # Proceed with pneumonia detection only if it's a valid X-ray
            with st.spinner('Analysing image...'):
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

            # Chat interface
            st.markdown("---")
            st.subheader("🤖 Clinical Assistant")

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