import google.generativeai as genai
from typing import Dict, Any, Optional
from PIL import Image


class ClinicalQA:
    """A medical AI assistant for interpreting chest X-ray analysis results using Gemini API.

    This class provides question-answering capabilities based on machine learning model outputs
    and GradCAM visualisations, maintaining strict adherence to provided data without speculation.

    Attributes:
        model (genai.GenerativeModel): Gemini model for text generation
        vision_model (genai.GenerativeModel): Gemini model for vision tasks
        analysis_context (dict): Stored analysis results and images
        chat (genai.ChatSession): Chat session with context
        system_prompt (str): Base prompt with instructions for responses
    """

    def __init__(self, api_key: str):
        """Initialise the ClinicalQA assistant with Gemini API configuration.

        Args:
            api_key (str): Google API key for accessing Gemini services
        """
        # Configure Gemini API with provided key
        genai.configure(api_key=api_key)
        
        # Initialise text and vision models
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.vision_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialise context and chat as None
        self.analysis_context = None
        self.chat = None

        # Define system prompt with strict guidelines
        self.system_prompt = """You are a medical AI assistant interpreting chest X-ray analysis results.  Your responses must:  
        - Use the provided machine learning model's classification as the primary diagnostic indicator  
        - Reference the GradCAM visualisation to explain model's focus areas  
        - Provide nuanced medical interpretation without making definitive diagnoses  
        - Explain medical implications of the model's prediction and confidence  
        - Always include appropriate medical disclaimers
        - You must not under any circumstances make your own judgements of the x-ray and must only rely on the classification, probability and heatmap location on the gradcam image
        - Never speculate on conditions or findings beyond what is explicitly indicated by the model's output and GradCAM visualisation
        - Do not provide specific anatomical locations unless they are clearly visible in the GradCAM visualisation
        - Focus on explaining what the model detected rather than inferring clinical conditions
        - Answer using British English
        Current analysis context: {context}  
        GradCAM instructions: Focus on explaining the highlighted areas in the heatmap  """

    def set_context(self, analysis_results: Dict[str, Any], original_image: Image.Image, 
                   gradcam_image: Optional[Image.Image] = None) -> None:
        """Store analysis results and images to provide context for questions.

        Args:
            analysis_results (Dict[str, Any]): Results from the pneumonia classifier
            original_image (Image.Image): Original chest X-ray image
            gradcam_image (Optional[Image.Image]): GradCAM visualisation image, if available
        """
        # Store context in a dictionary
        self.analysis_context = {
            'results': analysis_results,
            'original_image': original_image,
            'gradcam_image': gradcam_image
        }

        # Format detailed context string
        context_str = f"""  
        Machine Learning Model Results:  
        - Predicted Class: {analysis_results['class']}  
        - Prediction Confidence: {analysis_results['confidence']:.1f}%  
        - Probability Breakdown:  
        * Normal Chest: {analysis_results['probabilities']['Normal']:.1f}%  
        * Pneumonia Indication: {analysis_results['probabilities']['Pneumonia']:.1f}%  
        """

        # Start new chat session with formatted system prompt
        self.chat = self.model.start_chat(history=[])
        self.chat.send_message(self.system_prompt.format(context=context_str))

    def answer_question(self, question: str) -> str:
        """Generate a response to a user question based on stored context.

        Args:
            question (str): User's question about the chest X-ray analysis

        Returns:
            str: Response text or error message
        """
        # Check if context is set
        if self.analysis_context is None:
            return "Please upload and analyse an image first before asking questions."
        
        try:
            # Delegate to vision-based answering method
            return self.answer_question_with_image(question)
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def answer_question_with_image(self, question: str) -> str:
        """Generate a response using vision capabilities when image context is needed.

        Args:
            question (str): User's question about the chest X-ray analysis

        Returns:
            str: Response text based on model output and GradCAM, or error message
        """
        # Check if context is set
        if self.analysis_context is None:
            return "Please upload and analyse an image first before asking questions."
        
        try:
            # Prepare images for vision model
            images = []
            if self.analysis_context.get('gradcam_image') is not None:
                images.append(self.analysis_context['gradcam_image'])
            else:
                images.append(self.analysis_context['original_image'])
            
            # Get analysis results
            results = self.analysis_context['results']
            
            # Create detailed prompt with strict instructions
            prompt = f"""Chest X-ray Analysis:  
            Machine Learning Model Classification:  
            - Predicted Class: {results['class']}  
            - Confidence: {results['confidence']:.1f}%  
            - Probability Breakdown:  
            * Normal Chest: {results['probabilities']['Normal']:.1f}%  
            * Pneumonia Indication: {results['probabilities']['Pneumonia']:.1f}%  
            
            IMPORTANT INSTRUCTIONS:
            - Your response must be based ONLY on the GradCAM visualisation and the model's classification results
            - Do NOT make any clinical judgments beyond what the model explicitly indicates
            - Do NOT speculate on specific anatomical locations unless clearly visible in the GradCAM
            - Focus ONLY on explaining what the model detected rather than inferring clinical conditions
            - If you cannot answer based solely on the model's output and GradCAM, state that limitation
            
            Question: {question}"""
            
            # Generate response using vision model
            response = self.vision_model.generate_content([prompt] + images)
            return response.text
        
        except Exception as e:
            return f"Error generating response: {str(e)}"
        
    def generate_pneumonia_explanation(self) -> Optional[str]:
        """Generate an explanation for pneumonia classification based on model output.

        Returns:
            Optional[str]: Explanation text if pneumonia is detected and GradCAM is available,
                          None if not applicable, or error message if generation fails
        """
        # Check conditions for generating explanation
        if self.analysis_context is None or self.analysis_context['results']['class'] != "Pneumonia":
            return None

        results = self.analysis_context['results']
        
        if self.analysis_context.get('gradcam_image') is None:
            return "GradCAM visualisation not available for explanation."
        
        # Create prompt for pneumonia explanation
        prompt = f"""Chest X-ray Analysis:
        Machine Learning Model Classification:
        - Predicted Class: {results['class']}
        - Confidence: {results['confidence']:.1f}%
        - Probability: Pneumonia {results['probabilities']['Pneumonia']:.1f}%
        
        IMPORTANT INSTRUCTIONS:
        - Your response must be based ONLY on the GradCAM visualisation and the model's classification results
        - Do NOT make any clinical judgments beyond what the model explicitly indicates
        - Do NOT speculate on specific anatomical locations unless clearly visible in the GradCAM
        - Focus ONLY on explaining what the model detected rather than inferring clinical conditions
        - Provide a brief, factual description of the highlighted areas in the GradCAM visualisation
        - Do NOT diagnose specific types of pneumonia or suggest treatments
        
        Using the GradCAM heatmap, provide:
        1. A brief description of the highlighted areas in the visualisation
        2. A factual explanation of what these highlighted areas indicate about the model's decision
        Focus on the heatmap's highlighted regions and keep it concise."""

        try:
            # Generate response using vision model with GradCAM image
            response = self.vision_model.generate_content(
                [prompt, self.analysis_context['gradcam_image']]
            )
            return response.text
        except Exception as e:
            return f"Error generating explanation: {str(e)}"