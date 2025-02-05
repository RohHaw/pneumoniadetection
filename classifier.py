import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
from gradcam import GradCAM

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
        
        # Initialize GradCAM
        self.grad_cam = GradCAM(self.model, self.model.layer4[-1])
        
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
        
        # Generate GradCAM
        heatmap = self.grad_cam.generate(image_tensor)
        
        # Convert heatmap to RGB
        heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Convert original image to numpy array
        original_image = np.array(image)
        
        # Overlay heatmap on original image
        superimposed = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
            
        return {
            'class': self.classes[prediction.item()],
            'confidence': confidence.item() * 100,
            'probabilities': {
                'Normal': probabilities[0][0].item() * 100,
                'Pneumonia': probabilities[0][1].item() * 100
            },
            'gradcam': Image.fromarray(superimposed)
        }