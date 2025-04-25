"""
Module defining the PneumoniaClassifier for pneumonia detection.

This module implements a ResNet-50 based classifier for detecting pneumonia in chest X-ray images.
The classifier incorporates Monte Carlo Dropout for uncertainty quantification and EnhancedGradCAM
for visual explanations. It processes input images, generates predictions, and provides detailed
outputs including class probabilities, confidence intervals, Grad-CAM heatmaps, and bounding boxes.

Author: Rohman Hawrylak
Date: April 2025
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
from gradcam import EnhancedGradCAM

class PneumoniaClassifier(nn.Module):
    """
    A ResNet-50 based classifier for pneumonia detection with explainability features.

    This class implements a fine-tuned ResNet-50 model for classifying chest X-rays as normal or
    pneumonia-affected. It supports Monte Carlo Dropout for uncertainty estimation and integrates
    EnhancedGradCAM for generating visual explanations via heatmaps and bounding boxes.

    Attributes:
        device (torch.device): Device for computation (GPU or CPU).
        model (torchvision.models.ResNet): ResNet-50 model with custom fully connected layer.
        mc_dropout_iterations (int): Number of Monte Carlo Dropout iterations.
        enable_dropout (callable): Function to enable dropout during inference.
        grad_cam (EnhancedGradCAM): Instance for generating Grad-CAM heatmaps and boxes.
        transform (torchvision.transforms.Compose): Image preprocessing pipeline.
        classes (list): List of class labels ['Normal', 'Pneumonia'].
    """
    def __init__(self, model_path="Training/UCSD/model_UCSD_finetuned_rsna.pth", mc_dropout_iterations=20, dropout=0.3):
        """
        Initialise the PneumoniaClassifier with a pre-trained model and configuration.

        Args:
            model_path (str, optional): Path to pre-trained model weights. Defaults to
                "Training/UCSD/model_UCSD_finetuned_rsna.pth".
            mc_dropout_iterations (int, optional): Number of Monte Carlo Dropout iterations.
                Defaults to 20.
            dropout (float, optional): Dropout probability for the fully connected layer.
                Defaults to 0.3.
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

    def predict(self, image):
        """
        Predict pneumonia presence in an image with explainability outputs.

        Processes an input image using Monte Carlo Dropout to estimate prediction uncertainty and
        generates Grad-CAM heatmaps and bounding boxes for visual explanations. Returns a dictionary
        containing prediction details, confidence intervals, and visual outputs.

        Args:
            image (PIL.Image): Input chest X-ray image.

        Returns:
            dict: Prediction results including:
                - class (str): Predicted class ('Normal' or 'Pneumonia').
                - confidence (float): Confidence score in percentage.
                - uncertainty (float): Maximum standard deviation across classes in percentage.
                - probabilities (dict): Probabilities for 'Normal' and 'Pneumonia' in percentage.
                - confidence_interval (dict): 95% confidence intervals for each class.
                - gradcam (PIL.Image): Superimposed image with heatmap and bounding boxes.
                - heatmap_raw (numpy.ndarray): Resized raw heatmap.
                - boxes (list): List of bounding box dictionaries.
                - region_descriptions (list): Descriptions of highlighted regions.
        """
        original_width, original_height = image.size
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Enable dropout for Monte Carlo Dropout
        self.model.apply(self.enable_dropout)

        # Perform multiple forward passes for uncertainty estimation
        predictions = []
        with torch.no_grad():
            for _ in range(self.mc_dropout_iterations):
                outputs = self.model(image_tensor)
                prob = torch.nn.functional.softmax(outputs, dim=1)
                predictions.append(prob)

        # Compute mean and standard deviation of predictions
        predictions = torch.stack(predictions).cpu().numpy()
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)

        # Determine predicted class and confidence
        pred_class = np.argmax(mean_pred)
        confidence = mean_pred[0][pred_class]

        # Generate Grad-CAM heatmap and bounding boxes
        heatmap, boxes = self.grad_cam.generate_with_boxes(
            image_tensor,
            threshold=0.4,
            max_area_fraction=0.5,
            min_area_fraction=0.03,
            original_width=original_width,
            original_height=original_height
        )
        descriptions = self.grad_cam.get_region_descriptions(boxes)

        # Process heatmap for visualisation
        heatmap_resized = cv2.resize(heatmap, (original_width, original_height))
        heatmap_colored = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Create superimposed image with heatmap and bounding boxes
        original_image = np.array(image)
        superimposed = cv2.addWeighted(original_image, 0.6, heatmap_colored, 0.4, 0)

        for box in boxes:
            x, y, w, h = box['x'], box['y'], box['width'], box['height']
            cv2.rectangle(superimposed, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return {
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

    def to(self, device):
        """
        Move the model to the specified device.

        Args:
            device (torch.device): Target device (e.g., 'cuda' or 'cpu').

        Returns:
            PneumoniaClassifier: Self, for method chaining.
        """
        self.model.to(device)
        self.device = device
        return self

    def train(self, mode=True):
        """
        Set the model to training mode.

        Args:
            mode (bool, optional): If True, set to training mode; otherwise, evaluation mode.
                Defaults to True.
        """
        self.model.train(mode)

    def eval(self):
        """
        Set the model to evaluation mode.
        """
        self.model.eval()

    def parameters(self):
        """
        Return the model's parameters.

        Returns:
            Iterator: Iterator over the model's parameters.
        """
        return self.model.parameters()