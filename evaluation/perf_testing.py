"""
Module for performance testing of the pneumonia detection classifier.

This script defines a PneumoniaClassifier class that uses a ResNet-50 model for pneumonia detection
in chest X-ray images, incorporating Monte Carlo Dropout for uncertainty estimation and EnhancedGradCAM
for visual explanations. It includes a performance testing function to evaluate the classifier on a
set of images, measuring metrics such as processing time, resource usage, and prediction confidence.
Results are logged and saved to a CSV file for analysis. The script supports real-time evaluation on
a GPU or CPU and is designed for robustness with error handling.

Author: Rohman Hawrylak
Date: April 2025
"""

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
from gradcam import EnhancedGradCAM

# Configure logging to record performance metrics
logging.basicConfig(
    filename='pneumonia_classifier_performance_100.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

class PneumoniaClassifier(nn.Module):
    """
    A ResNet-50 based classifier for pneumonia detection with explainability features.

    This class implements a fine-tuned ResNet-50 model for classifying chest X-rays as normal or
    pneumonia-affected. It supports Monte Carlo Dropout for uncertainty quantification and integrates
    EnhancedGradCAM for visual explanations. The classifier processes images, generates predictions,
    and provides detailed performance metrics.

    Attributes:
        device (torch.device): Device for computation (GPU or CPU).
        model (torchvision.models.ResNet): ResNet-50 model with custom fully connected layer.
        mc_dropout_iterations (int): Number of Monte Carlo Dropout iterations.
        enable_dropout (callable): Function to enable dropout during inference.
        grad_cam (EnhancedGradCAM): Instance for generating Grad-CAM heatmaps and boxes.
        transform (torchvision.transforms.Compose): Image preprocessing pipeline.
        classes (list): List of class labels ['Normal', 'Pneumonia'].
    """
    def __init__(self, model_path="Training/UCSD/model_UCSD.pth", mc_dropout_iterations=20, dropout=0.5):
        """
        Initialise the PneumoniaClassifier with a pre-trained model and configuration.

        Args:
            model_path (str, optional): Path to pre-trained model weights. Defaults to
                "Training/UCSD/model_UCSD.pth".
            mc_dropout_iterations (int, optional): Number of Monte Carlo Dropout iterations.
                Defaults to 20.
            dropout (float, optional): Dropout probability for the fully connected layer.
                Defaults to 0.5.
        """
        super(PneumoniaClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialise ResNet-50 without pre-trained weights
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.model.fc.in_features, 2)
        )

        # Load pre-trained weights if available
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        
        self.model.to(self.device)
        self.model.eval()

        self.mc_dropout_iterations = mc_dropout_iterations

        # Define function to enable dropout during inference
        def enable_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()
        self.enable_dropout = enable_dropout

        # Initialise Grad-CAM for the last layer of ResNet-50
        self.grad_cam = EnhancedGradCAM(self.model, self.model.layer4[-1])

        # Define image preprocessing transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.classes = ['Normal', 'Pneumonia']

    def forward(self, x):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Model output logits.
        """
        return self.model(x)

    def predict_with_metrics(self, image):
        """
        Predict pneumonia presence in an image and collect performance metrics.

        Processes an input image, performs classification with Monte Carlo Dropout, generates Grad-CAM
        visualisations, and measures resource usage and processing times. Returns prediction results
        and detailed metrics.

        Args:
            image (PIL.Image): Input chest X-ray image.

        Returns:
            tuple: (results, metrics)
                - results (dict): Prediction details including class, confidence, uncertainty,
                    probabilities, confidence intervals, Grad-CAM outputs, and region descriptions.
                - metrics (dict): Performance metrics including CPU/GPU usage, memory, and processing
                    times.
        """
        metrics = {}

        # Record total processing time
        total_start_time = time.time()

        # Collect system resource usage
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory().percent
        gpu_memory_used = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2 if torch.cuda.is_available() else 0
        
        metrics['cpu_usage_percent'] = cpu_usage
        metrics['memory_usage_percent'] = memory
        metrics['gpu_memory_used_mb'] = gpu_memory_used
        metrics['gpu_memory_total_mb'] = gpu_memory_total

        # Preprocess the image
        preprocess_start = time.time()
        original_width, original_height = image.size
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        metrics['preprocess_time'] = time.time() - preprocess_start

        # Perform classification with Monte Carlo Dropout
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

        # Generate Grad-CAM heatmap and bounding boxes
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

        # Record total processing time
        metrics['total_time'] = time.time() - total_start_time

        # Store prediction details
        metrics['predicted_class'] = self.classes[pred_class]
        metrics['confidence_percent'] = confidence * 100
        metrics['normal_prob_percent'] = mean_pred[0][0] * 100
        metrics['pneumonia_prob_percent'] = mean_pred[0][1] * 100

        # Prepare results dictionary
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

        return results, metrics

def run_experiment(image_dir, num_images=100):
    """
    Run a performance experiment on multiple images and save metrics to a CSV file.

    Processes a specified number of images from a directory, evaluates the PneumoniaClassifier,
    and records performance metrics such as processing time, resource usage, and prediction details.
    Results are logged and saved to a CSV file for analysis. The function handles errors gracefully
    and provides progress updates.

    Args:
        image_dir (str): Directory containing input images.
        num_images (int, optional): Number of images to process. Defaults to 100.

    Returns:
        None: Outputs metrics to a CSV file and logs progress.
    """
    # Initialise the classifier
    print("Loading PneumoniaClassifier...")
    classifier = PneumoniaClassifier()

    # Collect supported image files
    supported_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(supported_extensions)]
    image_files = image_files[:min(num_images, len(image_files))]
    
    if not image_files:
        print(f"No valid images found in {image_dir}.")
        return

    # Store metrics for CSV output
    all_metrics = []

    # Process each image
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_dir, image_file)
        print(f"\nProcessing image {idx}/{len(image_files)}: {image_file}")
        
        try:
            image = Image.open(image_path).convert('RGB')
            start_time = time.time()
            results, metrics = classifier.predict_with_metrics(image)
            process_time = time.time() - start_time
            
            # Add image name and total processing time to metrics
            metrics['image_name'] = image_file
            metrics['total_process_time'] = process_time
            
            # Log key performance metrics
            logging.info(f"Image: {image_file}, Total Time: {metrics['total_time']:.3f}s, "
                        f"Classification: {metrics['classification_time']:.3f}s, "
                        f"GradCAM: {metrics['gradcam_time']:.3f}s, "
                        f"Class: {metrics['predicted_class']}, Confidence: {metrics['confidence_percent']:.1f}%")
            
            all_metrics.append(metrics)
            print(f"Completed {image_file} in {process_time:.3f} seconds")
        
        except Exception as e:
            logging.error(f"Error processing {image_file}: {str(e)}")
            print(f"Error processing {image_file}: {str(e)}")

    # Save metrics to CSV
    df = pd.DataFrame(all_metrics)
    csv_file = "pneumonia_classifier_metrics_100.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nExperiment complete. Metrics saved to {csv_file}")

if __name__ == "__main__":
    """
    Entry point for running the performance experiment.

    Configures the input directory and number of images, then executes the run_experiment function
    to evaluate the PneumoniaClassifier on a set of images.
    """
    # Configuration
    IMAGE_DIR = "/uolstore/home/users/sc21r2h/Documents/Year3/Dissertation/archive_combined/PNEUMONIA" 
    NUM_IMAGES = 100
    
    # Run the experiment
    run_experiment(IMAGE_DIR, NUM_IMAGES)