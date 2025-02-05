# clinical_qa.py
import google.generativeai as genai
from typing import Dict, Any
from PIL import Image

class ClinicalQA:
    def __init__(self, api_key: str):
        # Initialize Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.vision_model = genai.GenerativeModel('gemini-pro-vision')
        self.analysis_context = None
        self.chat = None
        
        # System prompt to set context
        self.system_prompt = """You are a medical AI assistant helping to interpret chest X-ray analysis results. 
        You should:
        - Provide clear, professional medical interpretations
        - Reference specific areas of the image and model predictions when relevant
        - Maintain appropriate medical terminology while being understandable
        - Always include appropriate medical disclaimers
        - Never make definitive diagnoses, instead describe observations and possibilities
        
        Current analysis context: {context}
        """

    def set_context(self, analysis_results: Dict[str, Any], original_image: Image.Image):
        """Store analysis results and image for context in answering questions"""
        self.analysis_context = {
            'results': analysis_results,
            'original_image': original_image,
        }
        
        # Format context for the system prompt
        context_str = f"""
        - Prediction: {analysis_results['class']}
        - Confidence: {analysis_results['confidence']:.1f}%
        - Probability Distribution: Normal ({analysis_results['probabilities']['Normal']:.1f}%), 
          Pneumonia ({analysis_results['probabilities']['Pneumonia']:.1f}%)
        """
        
        # Initialize new chat with context
        self.chat = self.model.start_chat(history=[])
        self.chat.send_message(self.system_prompt.format(context=context_str))

    def answer_question(self, question: str) -> str:
        """Generate a response using Gemini API"""
        if self.analysis_context is None:
            return "Please upload and analyze an image first before asking questions."
        
        try:
            # Send question to Gemini
            response = self.chat.send_message(question)
            return response.text
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
            
    async def answer_question_with_image(self, question: str) -> str:
        """Generate a response using Gemini Vision API when image context is needed"""
        if self.analysis_context is None:
            return "Please upload and analyze an image first before asking questions."
            
        try:
            # Prepare the image and results for Gemini
            image = self.analysis_context['original_image']
            results = self.analysis_context['results']
            
            # Create prompt with image context
            prompt = f"""Analyzing chest X-ray image with model results:
            Prediction: {results['class']}
            Confidence: {results['confidence']:.1f}%
            
            Question: {question}"""
            
            # Send to Gemini Vision API
            response = await self.vision_model.generate_content([prompt, image])
            return response.text
            
        except Exception as e:
            return f"Error generating response: {str(e)}"