import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
from gradcam import GradCAM

class PneumoniaClassifier:
    def __init__(self, model_path="best_model.pth", mc_dropout_iterations=20):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load ResNet-50 with modified FC layer
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.model.fc.in_features, 2)
        )

        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode

        self.mc_dropout_iterations = mc_dropout_iterations

        # Ensure dropout is enabled during MC Dropout inference
        def enable_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()

        self.enable_dropout = enable_dropout

        # GradCAM for interpretability
        self.grad_cam = GradCAM(self.model, self.model.layer4[-1])

        # Image transformations (ensure values match training dataset stats)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # Verify these match your training
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.classes = ['Normal', 'Pneumonia']

    def predict(self, image):
        """Predicts pneumonia probability and generates Grad-CAM heatmap."""
        # Transform input image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Enable Monte Carlo Dropout
        self.model.apply(self.enable_dropout)

        # Collect multiple predictions for uncertainty estimation
        predictions = []
        with torch.no_grad():
            for _ in range(self.mc_dropout_iterations):
                outputs = self.model(image_tensor)
                prob = torch.nn.functional.softmax(outputs, dim=1)  # Ensure softmax is applied only once
                predictions.append(prob)

        # Convert to NumPy arrays for easier processing
        predictions = torch.stack(predictions).cpu().numpy()
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)

        # Determine predicted class
        pred_class = np.argmax(mean_pred)
        confidence = mean_pred[0][pred_class]

        # Generate Grad-CAM heatmap
        heatmap = self.grad_cam.generate(image_tensor)

        # Resize heatmap to match original image size
        heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Convert original image to NumPy array
        original_image = np.array(image)

        # Overlay heatmap on the original image
        superimposed = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

        return {
            'class': self.classes[pred_class],
            'confidence': confidence * 100,
            'uncertainty': std_pred[0].max() * 100,  # Maximum uncertainty metric
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
            'gradcam': Image.fromarray(superimposed)
        }
