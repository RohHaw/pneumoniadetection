import streamlit as st
import numpy as np
from PIL import Image
from classifier import PneumoniaClassifier
from clinical_qa import ClinicalQA
import os

def modern_result_display(results, original_image, gradcam_image):
    """Create a modern, card-based result display"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
        <div style='background-color: #f0f2f6; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h3 style='color: #333; margin-bottom: 15px;'>🩺 Diagnosis</h3>
            <div style='text-align: center;'>
                <h2 style='color: {"red" if results["class"] == "Pneumonia" else "green"}; margin: 10px 0;'>
                    {"Potential Pneumonia Detected" if results["class"] == "Pneumonia" else "No Pneumonia Indication"}
                </h2>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background-color: #f0f2f6; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h3 style='color: #333; margin-bottom: 15px;'>📊 Pneumonia Probability</h3>
            <div style='text-align: center;'>
                <span style='color: red; font-weight: bold; font-size: 1.2em;'>Pneumonia Probability</span>
                <p style='margin: 10px 0; font-size: 1.5em; color: red;'>{results["probabilities"]["Pneumonia"]:.1f}%</p>
                <small style='color: #666;'>95% Confidence Interval:</small>
                <small style='color: #888;'>{results["confidence_interval"]["Pneumonia"][0]:.1f}% - {results["confidence_interval"]["Pneumonia"][1]:.1f}%</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # X-Ray Visualization
    st.markdown("---")
    st.subheader("🔬 X-Ray Analysis")
    
    # Side-by-side image display
    col_orig, col_grad = st.columns(2)
    
    with col_orig:
        st.image(original_image, caption="Original X-Ray", use_container_width=True)
    
    with col_grad:
        st.image(gradcam_image, caption="Model Focus Areas", use_container_width=True)

def main():
    st.set_page_config(page_title="Pneumonia X-Ray Classifier", layout="wide", page_icon="🩺")
    
    st.title("🩺 Pneumonia X-Ray Classification")
    
    # Get Gemini API key
    api_key = os.getenv('GEMINI_API_KEY') or st.secrets['GEMINI_API_KEY']
    
    # Initialize classifier and QA system
    try:
        classifier = PneumoniaClassifier()
        qa_system = ClinicalQA(api_key=api_key)
        model_load_success = True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        model_load_success = False
    
    if model_load_success:
        uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "jpeg", "png"], key="xray_uploader")
        
        if uploaded_file is not None:
            # Create columns for image and analysis
            col1, col2 = st.columns([2, 3])
            
            with col1:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded X-Ray Image", use_container_width=True)
            
            with col2:
                with st.spinner('Analyzing image...'):
                    try:
                        results = classifier.predict(image)
                        qa_system.set_context(results, image)
                        
                        # Modern result display with original and gradcam images
                        modern_result_display(results, image, results['gradcam'])
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
            
            # Chat Interface
            st.markdown("---")
            st.subheader("🤖 Clinical Assistant")
            
            # Initialize chat history
            if 'messages' not in st.session_state:
                st.session_state.messages = []
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask a question about the X-ray analysis"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Generating response..."):
                        if any(word in prompt.lower() for word in ['show', 'look', 'image', 'visual']):
                            response = qa_system.answer_question_with_image(prompt)
                        else:
                            response = qa_system.answer_question(prompt)
                        st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
        
        else:
            st.info("Upload a chest X-ray image to begin analysis")

if __name__ == "__main__":
    main()