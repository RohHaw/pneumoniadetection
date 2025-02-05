import google.generativeai as genai
from typing import Dict, Any
from PIL import Image

class ClinicalQA:
    def __init__(self, api_key: str):
        # Initialize Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.vision_model = genai.GenerativeModel('gemini-1.5-flash')
        self.analysis_context = None
        self.chat = None

        # Enhanced system prompt to emphasize using model's classification
        self.system_prompt = """You are a medical AI assistant interpreting chest X-ray analysis results.  Your responses must:  
        - Use the provided machine learning model's classification as the primary diagnostic indicator  
        - Reference the GradCAM visualization to explain model's focus areas  
        - Provide nuanced medical interpretation without making definitive diagnoses  
        - Explain medical implications of the model's prediction and confidence  
        - Always include appropriate medical disclaimers
        - You must not under any circumstances make your own judgements of the x-ray and must only rely on the classifcation, proability and heatmap location on the gradcam image. 
        Current analysis context: {context}  
        GradCAM instructions: Focus on explaining the highlighted areas in the heatmap  """

    def set_context(self, analysis_results: Dict[str, Any], original_image: Image.Image, gradcam_image: Image.Image = None):
        """Store analysis results, original image, and optional GradCAM image for context"""
        self.analysis_context = {
            'results': analysis_results,
            'original_image': original_image,
            'gradcam_image': gradcam_image
        }

        # Detailed context formatting
        context_str = f"""  
        Machine Learning Model Results:  
        - Predicted Class: {analysis_results['class']}  
        - Prediction Confidence: {analysis_results['confidence']:.1f}%  
        - Probability Breakdown:  
        * Normal Chest: {analysis_results['probabilities']['Normal']:.1f}%  
        * Pneumonia Indication: {analysis_results['probabilities']['Pneumonia']:.1f}%  
        """

        # Initialize new chat with context
        self.chat = self.model.start_chat(history=[])
        self.chat.send_message(self.system_prompt.format(context=context_str))

    def answer_question(self, question: str) -> str:
        """Generate a response using Gemini API"""
        if self.analysis_context is None:
            return "Please upload and analyze an image first before asking questions."
        
        try:
            response = self.chat.send_message(question)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def answer_question_with_image(self, question: str) -> str:
        """Generate a response using Gemini Vision API when image context is needed"""
        if self.analysis_context is None:
            return "Please upload and analyze an image first before asking questions."
        
        try:
            # Prepare the images and results for Gemini
            images = [self.analysis_context['original_image']]
            
            # Include GradCAM image if available
            if self.analysis_context.get('gradcam_image') is not None:
                images.append(self.analysis_context['gradcam_image'])
            
            results = self.analysis_context['results']
            
            # Create prompt emphasizing model's classification
            prompt = f"""Chest X-ray Analysis:  
            Machine Learning Model Classification:  
            - Predicted Class: {results['class']}  
            - Confidence: {results['confidence']:.1f}%  
            Analyze the image(s) with respect to the model's prediction and the GradCAM visualization.  
            Question: {question}"""
            
            # Send to Gemini Vision API
            response = self.vision_model.generate_content([prompt] + images)
            return response.text
        
        except Exception as e:
            return f"Error generating response: {str(e)}"