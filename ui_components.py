"""
UI components for displaying results and visualisations in the Pneumonia X-Ray Classifier.

This module provides Streamlit-based functions for rendering interactive analysis results,
visualisations, and analysis history. It includes components for displaying probability analyses,
GradCAM visualisations with bounding boxes, detailed classification results, a timeline of past
analyses, and a welcome screen for new users. The components are designed to be visually appealing
and user-friendly, with support for dark mode compatibility.

Author: Rohman Hawrylak
Date: April 2025
"""

import streamlit as st
import plotly.express as px
from PIL import Image
import numpy as np

def interactive_result_display(results, original_image, gradcam_image):
    """
    Display analysis results with text-based probability and image visualisations.

    Renders a tabbed interface with three sections: probability analysis, X-ray visualisation with
    GradCAM (for pneumonia cases), and detailed classification results. The probability section
    shows the pneumonia probability with a 95% confidence interval, the visualisation section
    displays the GradCAM heatmap with bounding boxes and region descriptions, and the detailed
    results section provides classification probabilities and confidence intervals.

    Args:
        results (dict): Dictionary containing class, probabilities, confidence intervals,
            GradCAM data, boxes, and region descriptions.
        original_image (PIL.Image): The original uploaded X-ray image.
        gradcam_image (PIL.Image): The GradCAM heatmap image with bounding boxes.
    """
    # Create tabs for different result views
    result_tabs = st.tabs(["📊 Probability Analysis", "🔬 X-Ray Visualisation", "📋 Detailed Results"])

    with result_tabs[0]:
        # Display pneumonia probability with 95% confidence interval
        pneumonia_prob = results["probabilities"]["Pneumonia"]
        ci_pneumonia = results["confidence_interval"]["Pneumonia"]
        
        # Calculate the ± margin from the 95% confidence interval
        lower_bound, upper_bound = ci_pneumonia[0], ci_pneumonia[1]
        margin = (upper_bound - lower_bound) / 2  # Half the CI width
        
        # Format the text with higher precision
        prob_text = f"Pneumonia Probability: {pneumonia_prob:.2f}% ± {margin:.2f}%"
        
        # Display with styled container
        st.markdown(
            f"""
            <div style='background-color: #F3F4F6; padding: 1rem; border-radius: 5px;'>
                <h3 style='color: #1F2937; margin: 0;'>{prob_text}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    with result_tabs[1]:
        # Display GradCAM visualisation only for pneumonia cases
        if results["class"] == "Pneumonia":
            gradcam_resized = gradcam_image.resize((550, 550))
            st.image(gradcam_resized, caption="GradCAM Visualisation - Highlighted Regions of Interest with Bounding Boxes", use_container_width=False)

            # Display region descriptions if available
            if "region_descriptions" in results and results["region_descriptions"]:
                st.markdown("**Highlighted Regions:**")
                for desc in results["region_descriptions"]:
                    st.write(f"- {desc}")

            # Provide explanation of GradCAM
            with st.expander("What is GradCAM?"):
                st.markdown("""
                **Gradient-weighted Class Activation Mapping (GradCAM)** highlights regions in the X-ray that contributed most to the AI's decision.
                - **Red areas** indicate strong relevance to the detected class (Pneumonia).
                - **Blue areas** have minimal influence.
                - **Green boxes** outline regions of significant interest identified by the model.
                
                For pneumonia detection, GradCAM typically highlights opacities or abnormal lung patterns.
                """)
        else:
            st.info("No GradCAM visualisation available for normal X-rays.")

    with result_tabs[2]:
        st.subheader("Classification Details")

        # Split layout into two columns for probabilities and confidence intervals
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Primary Classification:** {results['class']}")
            st.json({
                "Normal": f"{results['probabilities']['Normal']:.2f}%",
                "Pneumonia": f"{pneumonia_prob:.2f}%"
            })

        with col2:
            st.markdown("**Confidence Intervals (95%)**")
            ci_normal = results["confidence_interval"]["Normal"]
            ci_pneumonia = results["confidence_interval"]["Pneumonia"]

            st.json({
                "Normal": f"[{ci_normal[0]:.2f}%, {ci_normal[1]:.2f}%]",
                "Pneumonia": f"[{ci_pneumonia[0]:.2f}%, {ci_pneumonia[1]:.2f}%]"
            })

        # Provide interpretation guide for probabilities
        st.markdown("""
        **Interpretation Guide**  
        - **< 20% Pneumonia Probability**: Likely normal  
        - **20-50% Probability**: Uncertain, clinical correlation recommended  
        - **> 50% Probability**: Suspicious for pneumonia  
        - **> 80% Probability**: Highly suspicious for pneumonia  

        This AI assessment is intended to assist healthcare professionals and should not replace clinical judgment.
        """)

def display_history():
    """
    Display the analysis history in a visually appealing format.

    Renders a timeline of pneumonia probabilities for past analyses (if multiple entries exist)
    and displays individual analysis entries as cards with images, classification details, and
    timestamps. Each card includes a button to view detailed results, which updates the session
    state and navigates to the analysis tab.
    """
    # Check if history is empty
    if not st.session_state.analysis_history:
        st.info("No analysis history available")
        return
    
    # Display header with history count
    st.subheader(f"Recent Analyses ({len(st.session_state.analysis_history)})")
    
    # Extract data for timeline
    timestamps = [entry["timestamp"] for entry in st.session_state.analysis_history]
    classes = [entry["results"]["class"] for entry in st.session_state.analysis_history]
    probabilities = [entry["results"]["probabilities"]["Pneumonia"] for entry in st.session_state.analysis_history]
    
    # Display timeline if multiple entries exist
    if len(timestamps) > 1:
        fig = px.line(
            x=timestamps, 
            y=probabilities,
            markers=True,
            labels={"x": "Time", "y": "Pneumonia Probability (%)"}
        )
        
        # Customise timeline appearance
        fig.update_traces(
            marker=dict(
                size=12,
                color=["#FF4B4B" if p > 50 else "#4CAF50" for p in probabilities],
                line=dict(width=1, color="DarkSlateGrey")
            ),
            line=dict(color="#BFDBFE", width=2)
        )
        
        # Configure layout for readability
        fig.update_layout(
            title="Pneumonia Probability Timeline",
            yaxis_range=[0, 100],
            height=300,
            hovermode="x unified",
            plot_bgcolor="white",
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Display history entries in a grid of cards
    cols_per_row = 3
    for i in range(0, len(st.session_state.analysis_history), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i + j
            if idx < len(st.session_state.analysis_history):
                entry = st.session_state.analysis_history[idx]
                with cols[j]:
                    pneumonia_prob = entry["results"]["probabilities"]["Pneumonia"]
                    card_color = "#FEE2E2" if pneumonia_prob > 50 else "#DCFCE7"
                    text_color = "#DC2626" if pneumonia_prob > 50 else "#059669"
                    
                    # Display card with classification and timestamp
                    st.markdown(f"""
                    <div style='background-color: {card_color}; 
                              padding: 0.5rem; 
                              border-radius: 5px; 
                              margin-bottom: 1rem;'>
                        <p style='font-weight: bold; margin: 0; color: {text_color};'>
                            {entry["results"]["class"]} ({pneumonia_prob:.1f}%)
                        </p>
                        <p style='margin: 0; font-size: 0.8rem;'>
                            {entry["timestamp"].strftime("%Y-%m-%d %H:%M")}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display thumbnail of X-ray image
                    st.image(entry["image"], width=120)
                    
                    # Button to view detailed results
                    if st.button(f"View Details", key=f"view_{idx}"):
                        st.session_state.last_results = {
                            "results": entry["results"],
                            "image": entry["image"],
                            "timestamp": entry["timestamp"]
                        }
                        st.query_params(tab="analysis")

def show_welcome_screen():
    """
    Display a welcome screen with instructions for using the application.

    Renders a styled container with a welcome message, usage instructions, and supported file
    formats. Includes CSS for dark mode compatibility to ensure consistent appearance across
    themes.
    """
    st.write(
        """
        <div class="welcome-container" style='background-color: #EFF6FF; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; color: #1F2937;'>
            <h3 style='color: #1E40AF; margin-top: 0;'>Welcome to the Pneumonia X-Ray Classifier</h3>
            <p style='color: #1F2937;'>This tool uses advanced AI to analyse chest X-rays for signs of pneumonia.</p>
            <h4 style='color: #1E40AF;'>How to use:</h4>
            <ol style='color: #1F2937; margin-left: 1.5rem;'>
                <li>Upload a chest X-ray image using the file uploader above</li>
                <li>Click "Analyse X-Ray" to process the image</li>
                <li>Review the results and visualisation</li>
                <li>Ask the Clinical Assistant any questions about the analysis</li>
            </ol>
            <p style='color: #1F2937;'><b>Supported formats:</b> JPG, JPEG, PNG</p>
        <style>
            .welcome-container, .upload-prompt {
                color: inherit;
            }
            @media (prefers-color-scheme: dark) {
                .welcome-container {
                    background-color: #1E3A8A !important;
                    color: #E5E7EB !important;
                }
                .welcome-container h3, .welcome-container h4 {
                    color: #93C5FD !important;
                }
                .welcome-container p, .welcome-container ol, .welcome-container li {
                    color: #E5E7EB !important;
                }
                .upload-prompt {
                    color: #D1D5DB !important;
                }
                .upload-prompt p {
                    color: #D1D5DB !important;
                }
                .welcome-container p[style*="color: #1F2937"] {
                    color: #E5E7EB !important;
                }
            }
        </style>
        """,
        unsafe_allow_html=True
    )