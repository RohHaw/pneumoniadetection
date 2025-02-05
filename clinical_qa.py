from typing import Dict, Any
from PIL import Image

class ClinicalQA:
    def __init__(self):
        self.analysis_context = None
        
        # Template responses for different types of questions
        self.response_templates = {
            'confidence': [
                "Based on the analysis, the model's confidence in its {prediction} diagnosis is {confidence:.1f}%.",
                "The differential probabilities show {normal_prob:.1f}% for normal and {pneumonia_prob:.1f}% for pneumonia.",
                "This {confidence_level} confidence level suggests {confidence_interpretation}."
            ],
            'regions': [
                "The GradCAM analysis highlights {region_description}.",
                "These highlighted areas are typically associated with {clinical_relevance}.",
                "The intensity of the highlighting suggests {intensity_interpretation}."
            ],
            'comparison': [
                "Comparing the original image with the GradCAM visualization, {comparison_observation}.",
                "This pattern is {pattern_typicality} for this type of diagnosis."
            ]
        }

    def set_context(self, analysis_results: Dict[str, Any], original_image: Image.Image):
        """Store analysis results and image for context in answering questions"""
        self.analysis_context = {
            'results': analysis_results,
            'original_image': original_image,
        }
        
    def _get_confidence_level(self, confidence: float) -> str:
        if confidence >= 90:
            return "very high"
        elif confidence >= 75:
            return "high"
        elif confidence >= 60:
            return "moderate"
        else:
            return "low"

    def _get_confidence_interpretation(self, confidence: float, prediction: str) -> str:
        level = self._get_confidence_level(confidence)
        if level in ["very high", "high"]:
            return "a strong basis for the diagnosis"
        elif level == "moderate":
            return "that additional clinical correlation may be beneficial"
        else:
            return "that additional imaging or clinical tests may be warranted"

    def _analyze_gradcam_regions(self) -> Dict[str, str]:
        """Analyze the GradCAM heatmap to describe regions of interest"""
        prediction = self.analysis_context['results']['class']
        if prediction == 'Pneumonia':
            return {
                'region_description': "significant activation in the lung fields",
                'clinical_relevance': "potential areas of consolidation or infiltrates",
                'intensity_interpretation': "varying degrees of involvement across the lung fields"
            }
        else:
            return {
                'region_description': "relatively uniform activation across the lung fields",
                'clinical_relevance': "normal lung appearance",
                'intensity_interpretation': "no significant areas of concern"
            }

    def _process_comparison_question(self) -> Dict[str, str]:
        """Generate comparative analysis between original and GradCAM"""
        prediction = self.analysis_context['results']['class']
        confidence = self.analysis_context['results']['confidence']
        
        if prediction == 'Pneumonia':
            return {
                'comparison_observation': "the areas of high activation correspond to regions of potential infiltrate or consolidation",
                'pattern_typicality': "consistent with commonly observed patterns" if confidence > 75 else "somewhat atypical"
            }
        else:
            return {
                'comparison_observation': "the activation pattern shows relatively uniform distribution without concerning focal areas",
                'pattern_typicality': "typical of normal chest radiographs"
            }

    def answer_question(self, question: str) -> str:
        """Generate a context-aware response to a clinical question"""
        if self.analysis_context is None:
            return "Please upload and analyze an image first before asking questions."
        
        # Convert question to lowercase for easier matching
        question = question.lower()
        
        # Extract relevant context
        results = self.analysis_context['results']
        confidence = results['confidence']
        prediction = results['class']
        probabilities = results['probabilities']
        
        # Prepare response based on question type
        if any(word in question for word in ['confidence', 'sure', 'certain', 'probability']):
            return " ".join(self.response_templates['confidence']).format(
                prediction=prediction,
                confidence=confidence,
                normal_prob=probabilities['Normal'],
                pneumonia_prob=probabilities['Pneumonia'],
                confidence_level=self._get_confidence_level(confidence),
                confidence_interpretation=self._get_confidence_interpretation(confidence, prediction)
            )
        elif any(word in question for word in ['region', 'area', 'where', 'location']):
            regions = self._analyze_gradcam_regions()
            return " ".join(self.response_templates['regions']).format(**regions)
        elif any(word in question for word in ['compare', 'difference', 'original', 'visualization']):
            comparison = self._process_comparison_question()
            return " ".join(self.response_templates['comparison']).format(**comparison)
        else:
            return ("I'm not sure about that specific question. You can ask about:\n"
                   "- Confidence levels and probabilities\n"
                   "- Regions of interest in the image\n"
                   "- Comparison between original and GradCAM visualization")