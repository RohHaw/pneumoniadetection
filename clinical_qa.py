import google.generativeai as genai
from typing import Dict, Any
from PIL import Image

class ClinicalQA:
    def __init__(self, api_key: str):
        # Initialize Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
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
        - You must not under any circumstances make your own judgements of the x-ray and must only rely on the classification, probability and heatmap location on the gradcam image
        - Never speculate on conditions or findings beyond what is explicitly indicated by the model's output and GradCAM visualization
        - Do not provide specific anatomical locations unless they are clearly visible in the GradCAM visualization
        - Focus on explaining what the model detected rather than inferring clinical conditions
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
            # Always use the vision model with images for all questions
            return self.answer_question_with_image(question)
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def answer_question_with_image(self, question: str) -> str:
        """Generate a response using Gemini Vision API when image context is needed"""
        if self.analysis_context is None:
            return "Please upload and analyze an image first before asking questions."
        
        try:
            # Prepare the images and results for Gemini
            images = []
            
            # Always include GradCAM image if available
            if self.analysis_context.get('gradcam_image') is not None:
                images.append(self.analysis_context['gradcam_image'])
            else:
                # If no GradCAM, use the original image
                images.append(self.analysis_context['original_image'])
            
            results = self.analysis_context['results']
            
            # Create prompt emphasizing model's classification and restrictions
            prompt = f"""Chest X-ray Analysis:  
            Machine Learning Model Classification:  
            - Predicted Class: {results['class']}  
            - Confidence: {results['confidence']:.1f}%  
            - Probability Breakdown:  
            * Normal Chest: {results['probabilities']['Normal']:.1f}%  
            * Pneumonia Indication: {results['probabilities']['Pneumonia']:.1f}%  
            
            IMPORTANT INSTRUCTIONS:
            - Your response must be based ONLY on the GradCAM visualization and the model's classification results
            - Do NOT make any clinical judgments beyond what the model explicitly indicates
            - Do NOT speculate on specific anatomical locations unless clearly visible in the GradCAM
            - Focus ONLY on explaining what the model detected rather than inferring clinical conditions
            - If you cannot answer based solely on the model's output and GradCAM, state that limitation
            
            Question: {question}"""
            
            # Send to Gemini Vision API
            response = self.vision_model.generate_content([prompt] + images)
            return response.text
        
        except Exception as e:
            return f"Error generating response: {str(e)}"
        
    def generate_pneumonia_explanation(self) -> str:
        """Generate a description and explanation for pneumonia classification."""
        if self.analysis_context is None or self.analysis_context['results']['class'] != "Pneumonia":
            return None

        results = self.analysis_context['results']
        
        if self.analysis_context.get('gradcam_image') is None:
            return "GradCAM visualization not available for explanation."
        
        prompt = f"""Chest X-ray Analysis:
        Machine Learning Model Classification:
        - Predicted Class: {results['class']}
        - Confidence: {results['confidence']:.1f}%
        - Probability: Pneumonia {results['probabilities']['Pneumonia']:.1f}%
        
        IMPORTANT INSTRUCTIONS:
        - Your response must be based ONLY on the GradCAM visualization and the model's classification results
        - Do NOT make any clinical judgments beyond what the model explicitly indicates
        - Do NOT speculate on specific anatomical locations unless clearly visible in the GradCAM
        - Focus ONLY on explaining what the model detected rather than inferring clinical conditions
        - Provide a brief, factual description of the highlighted areas in the GradCAM visualization
        - Do NOT diagnose specific types of pneumonia or suggest treatments
        
        Using the GradCAM heatmap, provide:
        1. A brief description of the highlighted areas in the visualization
        2. A factual explanation of what these highlighted areas indicate about the model's decision
        Focus on the heatmap's highlighted regions and keep it concise."""

        try:
            response = self.vision_model.generate_content(
                [prompt, self.analysis_context['gradcam_image']]
            )
            return response.text
        except Exception as e:
            return f"Error generating explanation: {str(e)}"