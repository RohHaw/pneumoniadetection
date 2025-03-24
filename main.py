"""Main Streamlit application for Pneumonia X-Ray Classification."""

import streamlit as st
from PIL import Image
import os
from datetime import datetime
import time
from config import SUPPORTED_FILE_TYPES, MAX_FILE_SIZE, CONFIDENCE_THRESHOLD, HISTORY_LIMIT
from utils import (
    initialise_session_state, reset_session_state, load_models, generate_results_csv, limit_history
)
from ui_components import interactive_result_display, display_history, show_welcome_screen
from clinical_qa import ClinicalQA

def main():
    """Run the Pneumonia X-Ray Classifier Streamlit app."""
    st.set_page_config(
        page_title="Pneumonia X-Ray Classifier", 
        layout="wide", 
        page_icon="🩺",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton>button {
            background-color: #4F8BF9;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        .stAlert {
            border-radius: 5px;
        }
        .stImage {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        h1, h2, h3 {
            color: #1E3A8A;
        }
        p, li, ol, ul {
            color: #1F2937;
        }
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        .css-1aumxhk {
            background-color: #f1f5f9;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        /* Dark mode compatibility */
        @media (prefers-color-scheme: dark) {
            p, li, ol, ul {
                color: #E5E7EB;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #93C5FD;
            }
            .stMarkdown div {
                color: #E5E7EB;
            }
            p[style*="color: #1F2937"] {
                color: #E5E7EB !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)
    
    initialise_session_state()

    # Application header
    with st.container():
        col1, col2 = st.columns([1, 5])
        with col1:
            st.markdown("")
        with col2:
            st.title("Pneumonia X-Ray Classifier")
            st.markdown("""
            <p style='font-size: 1.1rem; color: #FFFFFF;'>
                Advanced AI-powered chest X-ray analysis for pneumonia detection with clinical insights
            </p>
            """, unsafe_allow_html=True)
    
    st.markdown("---")

    # Load models and API key
    with st.spinner("Loading AI models..."):
        xray_validator, classifier = load_models()
        api_key = os.getenv('GEMINI_API_KEY') or st.secrets.get('GEMINI_API_KEY', None)
        qa_system = ClinicalQA(api_key=api_key) if api_key else None
        
        # Restore QA context if it exists
        if qa_system and 'qa_context' in st.session_state:
            qa_system.set_context(
                analysis_results=st.session_state.qa_context['analysis_results'],
                original_image=st.session_state.qa_context['original_image'],
                gradcam_image=st.session_state.qa_context['gradcam_image']
            )

    if not xray_validator or not classifier:
        st.error("⚠️ Model loading failed. Please check the logs and try again.")
        return

    # Create tabs for different functions
    tab1, tab2, tab3 = st.tabs(["📊 Analysis", "📚 History", "ℹ️ About"])

    with tab1:
        # Left column for upload and right for results
        left_col, right_col = st.columns([2, 3])
        
        with left_col:
            st.subheader("Upload X-Ray Image")
            
            uploaded_file = st.file_uploader(
                "Drop your chest X-ray here",
                type=SUPPORTED_FILE_TYPES,
                help=f"Upload a chest X-ray image (Supported formats: {', '.join(SUPPORTED_FILE_TYPES)})"
            )
            
            if uploaded_file:
                if uploaded_file.size > MAX_FILE_SIZE:
                    st.error(f"📁 File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)}MB.")
                else:
                    try:
                        image = Image.open(uploaded_file).convert('RGB')
                        image = image.resize((224, 224), Image.LANCZOS)
                        display_image = image.copy()
                        st.image(display_image, caption="Uploaded X-Ray", use_container_width=True)
                        
                        analyze_button = st.button(
                            "🔍 Analyse X-Ray",
                            type="primary",
                            use_container_width=True
                        )
                        
                        if analyze_button:
                            if 'last_results' in st.session_state:
                                del st.session_state.last_results
                                
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            status_text.text("Step 1/3: Validating X-ray...")
                            time.sleep(0.5)
                            progress_bar.progress(33)
                            
                            is_xray, confidence = xray_validator.validate_image(image)
                            
                            if not is_xray:
                                st.error(f"❌ The uploaded image does not appear to be a chest X-ray (confidence: {confidence:.1f}%).")
                                progress_bar.empty()
                                status_text.empty()
                                return
                                
                            if confidence < CONFIDENCE_THRESHOLD:
                                st.warning(f"⚠️ Low confidence ({confidence:.1f}%) that this is a chest X-ray. Results may be unreliable.")
                            
                            status_text.text("Step 2/3: Analysing for pneumonia...")
                            time.sleep(0.5)
                            progress_bar.progress(66)
                            
                            try:
                                results = classifier.predict(image)
                                
                                status_text.text("Step 3/3: Processing results...")
                                time.sleep(0.5)
                                progress_bar.progress(100)
                                
                                st.session_state.last_results = {
                                    "results": results,
                                    "image": display_image,
                                    "timestamp": datetime.now()
                                }
                                
                                if qa_system:
                                    qa_system.set_context(
                                        analysis_results=results,
                                        original_image=display_image,
                                        gradcam_image=results["gradcam"]
                                    )
                                    st.session_state.qa_context = {
                                        'analysis_results': results,
                                        'original_image': display_image,
                                        'gradcam_image': results["gradcam"]
                                    }
                                
                                pneumonia_explanation = None
                                if results["class"] == "Pneumonia" and qa_system:
                                    with st.spinner("Generating clinical explanation..."):
                                        pneumonia_explanation = qa_system.generate_pneumonia_explanation()
                                
                                history_entry = {
                                    "timestamp": datetime.now(),
                                    "results": results,
                                    "image": display_image
                                }
                                
                                if pneumonia_explanation:
                                    history_entry["explanation"] = pneumonia_explanation
                                    
                                st.session_state.analysis_history.append(history_entry)
                                limit_history(HISTORY_LIMIT)
                                
                                time.sleep(0.5)
                                progress_bar.empty()
                                status_text.empty()
                                
                                csv = generate_results_csv(results)
                                st.download_button(
                                    "📥 Download Results CSV",
                                    csv,
                                    "pneumonia_results.csv",
                                    "text/csv",
                                    key="download_results",
                                    use_container_width=True
                                )
                                
                            except Exception as e:
                                st.error(f"❌ Error during prediction: {str(e)}")
                                progress_bar.empty()
                                status_text.empty()
                                return
                            
                    except Exception as e:
                        st.error(f"❌ Failed to process the uploaded file: {str(e)}")
            else:
                show_welcome_screen()
        
        with right_col:
            if 'last_results' in st.session_state:
                st.subheader("Analysis Results")
                results = st.session_state.last_results["results"]
                pneu_prob = results["probabilities"]["Pneumonia"]

                if results["class"] == "Pneumonia":
                    st.markdown(f"""
                    <div style='background-color: #FEE2E2; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;'>
                        <h3 style='color: #DC2626; margin: 0;'>Pneumonia Detected</h3>
                        <p style='color: #7F1D1D; margin: 0;'>Probability: {pneu_prob:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    last_item = st.session_state.analysis_history[-1]
                    if "explanation" in last_item and last_item["explanation"]:
                        explanation = last_item["explanation"]
                        if not explanation.startswith("Error") and not explanation.startswith("GradCAM"):
                            st.markdown("**Clinical Explanation:**")
                            st.write(explanation)
                        else:
                            st.warning("Could not generate clinical explanation due to an error.")
                else:
                    st.markdown(f"""
                    <div style='background-color: #DCFCE7; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;'>
                        <h3 style='color: #059669; margin: 0;'>Normal X-Ray</h3>
                        <p style='color: #065F46; margin: 0;'>Probability: {100-pneu_prob:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                interactive_result_display(
                    st.session_state.last_results["results"],
                    st.session_state.last_results["image"],
                    st.session_state.last_results["results"]["gradcam"]
                )
                
                st.markdown(f"""
                <div style='margin-top: 1rem; color: #6B7280; font-size: 0.8rem;'>
                    Analysis completed at {st.session_state.last_results["timestamp"].strftime('%Y-%m-%d %H:%M:%S')}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("👈 Upload and analyse an X-ray to see results here")
    
    with tab2:
        if st.session_state.analysis_history:
            display_history()
        else:
            st.info("No analysis history available yet. Analyse an X-ray to begin.")
    
    with tab3:
        st.subheader("About this Application")
        st.markdown("""
            This application uses deep learning to analyse chest X-rays for signs of pneumonia. 
            
            <h4 style='color: #1E40AF;'>How to use:</h4>
            <ol style='color: #1F2937; margin-left: 1.5rem;'>
                <li>Upload a chest X-ray image using the file uploader above</li>
                <li>Click "Analyse X-Ray" to process the image</li>
                <li>Review the results and visualisation</li>
                <li>Ask the Clinical Assistant any questions about the analysis</li>
            </ol>
            <p style='color: #FFFFFF;'><b>Supported formats:</b> JPG, JPEG, PNG</p>
        """, unsafe_allow_html=True)

    # Clinical Assistant Section
    st.markdown("---")
    st.subheader("🤖 Clinical Assistant")
    
    if not qa_system:
        st.warning("💡 Clinical QA is unavailable due to missing API key. Configure the GEMINI_API_KEY to enable this feature.")
    else:
        if 'qa_context' not in st.session_state:
            st.info("👆 Please upload and analyse an X-ray first to use the Clinical Assistant.")
        else:
            st.markdown("Ask questions about pneumonia, chest X-rays, or the analysis results")
            
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            if prompt := st.chat_input("Type your question here..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Generating response..."):
                        response = qa_system.answer_question(prompt)
                        st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    with st.sidebar:
        st.header("Options")
        
        if st.button("🔄 Reset Analysis", use_container_width=True):
            reset_session_state()
            st.success("✅ Analysis history and chat reset successfully.")
            
        st.markdown("---")
        
        st.subheader("Settings")
        st.markdown(f"**Confidence Threshold**: {CONFIDENCE_THRESHOLD}%")
        st.markdown(f"**History Limit**: {HISTORY_LIMIT} analyses")
        
        st.markdown("---")
        st.caption("Pneumonia X-Ray Classifier v1.0")
        st.caption("© 2025 Rohman Hawrylak")

if __name__ == "__main__":
    main()