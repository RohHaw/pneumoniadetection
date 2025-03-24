import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
from gradcam import EnhancedGradCAM  # Assuming this is your gradcam.py

class PneumoniaClassifier(nn.Module):  # Inherit from nn.Module
    def __init__(self, model_path="best_model_final.pth", mc_dropout_iterations=20, dropout=0.5):
        super(PneumoniaClassifier, self).__init__()  # Initialize nn.Module
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load ResNet-50 with modified FC layer
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout),  # Use configurable dropout
            nn.Linear(self.model.fc.in_features, 2)
        )

        # Load trained weights if provided
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode by default

        self.mc_dropout_iterations = mc_dropout_iterations

        # Ensure dropout is enabled during MC Dropout inference
        def enable_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()

        self.enable_dropout = enable_dropout

        # GradCAM for interpretability
        self.grad_cam = EnhancedGradCAM(self.model, self.model.layer3[-1])

        # Image transformations (ensure values match training dataset stats)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.classes = ['Normal', 'Pneumonia']

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def predict(self, image):
        """Predicts pneumonia probability and generates Enhanced Grad-CAM heatmap with boxes."""
        original_width, original_height = image.size  # Get original DICOM size
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

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

        heatmap, boxes = self.grad_cam.generate_with_boxes(
            image_tensor, 
            threshold=0.4,  # Lowered to capture more regions
            max_area_fraction=0.5,  # Allow larger boxes
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

    # Delegate methods to self.model
    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    def train(self, mode=True):
        self.model.train(mode)

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()