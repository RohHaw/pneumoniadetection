import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from classifier import PneumoniaClassifier
from clinical_qa import ClinicalQA

def main():
    st.set_page_config(page_title="Pneumonia X-Ray Classifier", layout="wide")
    
    st.title("Pneumonia X-Ray Classification with Clinical Q&A")
    
    # Initialize classifier and QA system
    try:
        classifier = PneumoniaClassifier()
        qa_system = ClinicalQA()
        model_load_success = True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        model_load_success = False
    
    if model_load_success:
        uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Create tabs for different views
            tab1, tab2 = st.tabs(["Analysis", "Clinical Q&A"])
            
            with tab1:
                col1, col2, col3 = st.columns(3)
                
                # Display uploaded image and get predictions
                image = Image.open(uploaded_file).convert('RGB')
                col1.subheader("Original Image")
                col1.image(image, use_column_width=True)
                
                with st.spinner('Analyzing image...'):
                    try:
                        results = classifier.predict(image)
                        
                        # Update QA system context
                        qa_system.set_context(results, image)
                        
                        # Display GradCAM
                        col2.subheader("GradCAM Visualization")
                        col2.image(results['gradcam'], use_column_width=True)
                        col2.info("Areas highlighted in red show regions the model focused on for its prediction")
                        
                        # Display results in col3
                        col3.subheader("Classification Results")
                        result_color = "green" if results['class'] == 'Normal' else "red"
                        col3.markdown(f"""
                        ### Diagnosis
                        <div style='padding: 20px; border-radius: 5px; background-color: rgba({result_color=="red" and "255,0,0,0.1" or "0,255,0,0.1"})'>
                            <h2 style='color: {result_color}; margin: 0;'>{results['class']}</h2>
                            <p style='font-size: 1.2em; margin: 10px 0;'>Confidence: {results['confidence']:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show probability distribution
                        col3.subheader("Probability Distribution")
                        fig = go.Figure(data=[
                            go.Bar(
                                x=list(results['probabilities'].keys()),
                                y=list(results['probabilities'].values()),
                                marker_color=['green', 'red']
                            )
                        ])
                        fig.update_layout(
                            title="Classification Probabilities",
                            yaxis_title="Probability (%)",
                            yaxis_range=[0, 100]
                        )
                        col3.plotly_chart(fig, use_container_width=True)
                        
                        # Add disclaimer
                        st.warning("""
                        **Disclaimer**: This tool is for educational purposes only. 
                        Medical diagnoses should always be made by qualified healthcare professionals.
                        """)
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
            
            with tab2:
                st.subheader("Clinical Q&A Interface")
                st.write("""
                Ask questions about the analysis results. For example:
                - How confident is the model in its diagnosis?
                - Which regions of the image influenced the decision?
                - How does the GradCAM visualization compare to the original image?
                """)
                
                # Question input
                question = st.text_input("Enter your question:")
                if question:
                    response = qa_system.answer_question(question)
                    st.write("Response:", response)
                
                # Chat history
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                
                if question:
                    st.session_state.chat_history.append(("Q: " + question, "A: " + response))
                
                # Display chat history
                if st.session_state.chat_history:
                    st.subheader("Previous Questions & Answers")
                    for q, a in reversed(st.session_state.chat_history):
                        st.text(q)
                        st.text(a)
                        st.markdown("---")
        
        else:
            st.info("Upload an X-ray image to begin analysis and ask questions.")

if __name__ == "__main__":
    main()