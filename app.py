import streamlit as st
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import io
import numpy as np

class PneumoniaClassifier:
    def __init__(self, model_path="best_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.model.fc.in_features, 2)
        )
        
        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Class labels
        self.classes = ['Normal', 'Pneumonia']

    def predict(self, image):
        # Transform image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            
        return {
            'class': self.classes[prediction.item()],
            'confidence': confidence.item() * 100,
            'probabilities': {
                'Normal': probabilities[0][0].item() * 100,
                'Pneumonia': probabilities[0][1].item() * 100
            }
        }

def main():
    st.set_page_config(page_title="Pneumonia X-Ray Classifier", layout="wide")
    
    st.title("Pneumonia X-Ray Classification")
    st.write("""
    Upload a chest X-ray image to check for pneumonia.
    The model will classify the image and provide confidence scores.
    """)
    
    # Initialize classifier
    try:
        classifier = PneumoniaClassifier()
        model_load_success = True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        model_load_success = False
        
    if model_load_success:
        # File uploader
        uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
        
        col1, col2 = st.columns(2)
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            col1.subheader("Uploaded Image")
            col1.image(image, use_column_width=True)
            
            # Make prediction
            with st.spinner('Analyzing image...'):
                try:
                    results = classifier.predict(image)
                    
                    # Display results
                    col2.subheader("Classification Results")
                    
                    # Create a nice-looking result display
                    result_color = "green" if results['class'] == 'Normal' else "red"
                    col2.markdown(f"""
                    ### Diagnosis
                    <div style='padding: 20px; border-radius: 5px; background-color: rgba({result_color=="red" and "255,0,0,0.1" or "0,255,0,0.1"})'>
                        <h2 style='color: {result_color}; margin: 0;'>{results['class']}</h2>
                        <p style='font-size: 1.2em; margin: 10px 0;'>Confidence: {results['confidence']:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show probability distribution
                    col2.subheader("Probability Distribution")
                    prob_df = {
                        'Condition': ['Normal', 'Pneumonia'],
                        'Probability (%)': [
                            results['probabilities']['Normal'],
                            results['probabilities']['Pneumonia']
                        ]
                    }
                    
                    # Create a bar chart
                    import plotly.graph_objects as go
                    fig = go.Figure(data=[
                        go.Bar(
                            x=prob_df['Condition'],
                            y=prob_df['Probability (%)'],
                            marker_color=['green', 'red']
                        )
                    ])
                    fig.update_layout(
                        title="Classification Probabilities",
                        yaxis_title="Probability (%)",
                        yaxis_range=[0, 100]
                    )
                    col2.plotly_chart(fig, use_container_width=True)
                    
                    # Add disclaimer
                    st.warning("""
                    **Disclaimer**: This tool is for educational purposes only. 
                    Medical diagnoses should always be made by qualified healthcare professionals.
                    """)
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
        
        else:
            # Show sample images and instructions when no image is uploaded
            st.info("""
            👆 Upload a chest X-ray image to get started.
            
            Make sure:
            - The image is clear and well-centered
            - It's a front-view (PA/AP) chest X-ray
            - The image is in JPG, JPEG, or PNG format
            """)

if __name__ == "__main__":
    main()