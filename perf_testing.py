import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import time
import logging
import psutil
import pandas as pd
from gradcam import EnhancedGradCAM  # Assuming this is in gradcam.py
from clinical_qa import ClinicalQA  # Assuming this is in clinical_qa.py

# Set up logging to a file for debugging (optional)
logging.basicConfig(
    filename='pneumonia_classifier_performance.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

class PneumoniaClassifier(nn.Module):
    def __init__(self, model_path="Training/UCSD/model_UCSD.pth", mc_dropout_iterations=20, dropout=0.5):
        super(PneumoniaClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load ResNet-50 with no pretrained weights
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.model.fc.in_features, 2)
        )

        # Load trained weights
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        
        self.model.to(self.device)
        self.model.eval()

        self.mc_dropout_iterations = mc_dropout_iterations

        # Enable dropout during inference
        def enable_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()
        self.enable_dropout = enable_dropout

        # GradCAM
        self.grad_cam = EnhancedGradCAM(self.model, self.model.layer3[-1])

        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.classes = ['Normal', 'Pneumonia']

    def forward(self, x):
        return self.model(x)

    def predict_with_metrics(self, image, qa_system=None):
        """Predict pneumonia with detailed performance metrics."""
        metrics = {}

        # Start total timing
        total_start_time = time.time()

        # System resources
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory().percent
        gpu_memory_used = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2 if torch.cuda.is_available() else 0
        
        metrics['cpu_usage_percent'] = cpu_usage
        metrics['memory_usage_percent'] = memory
        metrics['gpu_memory_used_mb'] = gpu_memory_used
        metrics['gpu_memory_total_mb'] = gpu_memory_total

        # Preprocessing
        preprocess_start = time.time()
        original_width, original_height = image.size
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        metrics['preprocess_time'] = time.time() - preprocess_start

        # Classification (Inference with MC Dropout)
        classification_start = time.time()
        self.model.apply(self.enable_dropout)
        predictions = []
        with torch.no_grad():
            for _ in range(self.mc_dropout_iterations):
                outputs = self.model(image_tensor)
                prob = torch.nn.functional.softmax(outputs, dim=1)
                predictions.append(prob)
        predictions = torch.stack(predictions).cpu().numpy()
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        pred_class = np.argmax(mean_pred)
        confidence = mean_pred[0][pred_class]
        metrics['classification_time'] = time.time() - classification_start

        # GradCAM
        gradcam_start = time.time()
        heatmap, boxes = self.grad_cam.generate_with_boxes(
            image_tensor,
            threshold=0.4,
            max_area_fraction=0.5,
            min_area_fraction=0.03,
            original_width=original_width,
            original_height=original_height
        )
        descriptions = self.grad_cam.get_region_descriptions(boxes)
        heatmap_resized = cv2.resize(heatmap, (original_width, original_height))
        heatmap_colored = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        original_image = np.array(image)
        superimposed = cv2.addWeighted(original_image, 0.6, heatmap_colored, 0.4, 0)
        for box in boxes:
            x, y, w, h = box['x'], box['y'], box['width'], box['height']
            cv2.rectangle(superimposed, (x, y), (x + w, y + h), (0, 255, 0), 2)
        metrics['gradcam_time'] = time.time() - gradcam_start

        # Prepare results
        results = {
            'class': self.classes[pred_class],
            'confidence': confidence * 100,
            'uncertainty': std_pred[0].max() * 100,
            'probabilities': {
                'Normal': mean_pred[0][0] * 100,
                'Pneumonia': mean_pred[0][1] * 100
            },
            'confidence_interval': {
                'Normal': (
                    mean_pred[0][0] * 100 - 1.96 * std_pred[0][0] * 100,
                    mean_pred[0][0] * 100 + 1.96 * std_pred[0][0] * 100
                ),
                'Pneumonia': (
                    mean_pred[0][1] * 100 - 1.96 * std_pred[0][1] * 100,
                    mean_pred[0][1] * 100 + 1.96 * std_pred[0][1] * 100
                )
            },
            'gradcam': Image.fromarray(superimposed),
            'heatmap_raw': heatmap_resized,
            'boxes': boxes,
            'region_descriptions': descriptions
        }

        # ClinicalQA text generation (if pneumonia detected and QA system provided)
        metrics['clinicalqa_time'] = 0.0
        metrics['clinicalqa_success'] = False
        if qa_system and results['class'] == 'Pneumonia':
            qa_system.set_context(results, image, results['gradcam'])
            clinicalqa_start = time.time()
            explanation = qa_system.generate_pneumonia_explanation()
            metrics['clinicalqa_time'] = time.time() - clinicalqa_start
            metrics['clinicalqa_success'] = not explanation.startswith("Error") if explanation else False

        # Total time
        metrics['total_time'] = time.time() - total_start_time

        # Prediction details
        metrics['predicted_class'] = self.classes[pred_class]
        metrics['confidence_percent'] = confidence * 100
        metrics['normal_prob_percent'] = mean_pred[0][0] * 100
        metrics['pneumonia_prob_percent'] = mean_pred[0][1] * 100

        return results, metrics

def run_experiment(image_dir, num_images=10, api_key=None):
    """Run an automated experiment on multiple images and save metrics to CSV."""
    # Load classifier
    print("Loading PneumoniaClassifier...")
    classifier = PneumoniaClassifier()

    # Load ClinicalQA if API key is provided
    qa_system = ClinicalQA(api_key=api_key) if api_key else None
    if not qa_system:
        print("Warning: ClinicalQA not initialized due to missing API key.")

    # Get list of image files
    supported_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(supported_extensions)]
    image_files = image_files[:min(num_images, len(image_files))]
    
    if not image_files:
        print(f"No valid images found in {image_dir}.")
        return

    # Data storage for CSV
    all_metrics = []

    # Process each image
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_dir, image_file)
        print(f"\nProcessing image {idx}/{len(image_files)}: {image_file}")
        
        try:
            image = Image.open(image_path).convert('RGB')
            start_time = time.time()
            results, metrics = classifier.predict_with_metrics(image, qa_system)
            process_time = time.time() - start_time
            
            # Add image name and total process time to metrics
            metrics['image_name'] = image_file
            metrics['total_process_time'] = process_time
            
            # Log key metrics
            logging.info(f"Image: {image_file}, Total Time: {metrics['total_time']:.3f}s, "
                        f"Classification: {metrics['classification_time']:.3f}s, "
                        f"GradCAM: {metrics['gradcam_time']:.3f}s, "
                        f"ClinicalQA: {metrics['clinicalqa_time']:.3f}s, "
                        f"Class: {metrics['predicted_class']}, Confidence: {metrics['confidence_percent']:.1f}%")
            
            all_metrics.append(metrics)
            print(f"Completed {image_file} in {process_time:.3f} seconds")
        
        except Exception as e:
            logging.error(f"Error processing {image_file}: {str(e)}")
            print(f"Error processing {image_file}: {str(e)}")

    # Save metrics to CSV
    df = pd.DataFrame(all_metrics)
    csv_file = "pneumonia_classifier_metrics.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nExperiment complete. Metrics saved to {csv_file}")

if __name__ == "__main__":
    # Configuration
    IMAGE_DIR = "/uolstore/home/users/sc21r2h/Documents/Year3/Dissertation/archive_combined/PNEUMONIA" 
    NUM_IMAGES = 10
    API_KEY = os.getenv('GEMINI_API_KEY')  # Set your Gemini API key here or in environment variable
    
    # Run the experiment
    run_experiment(IMAGE_DIR, NUM_IMAGES, API_KEY)