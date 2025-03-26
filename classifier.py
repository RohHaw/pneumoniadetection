import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
from gradcam import EnhancedGradCAM

class PneumoniaClassifier(nn.Module):
    def __init__(self, model_path="Training/best_rsna_model.pth", mc_dropout_iterations=20, 
                 dropout=0.3, target_layer="layer4[-1]", input_size=224, 
                 threshold=0.4, max_area_fraction=0.5, min_area_fraction=0.03, 
                 use_morph_ops=True, use_gradcam_plus_plus=False):
        super(PneumoniaClassifier, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(weights=None)  # Updated to use weights=None instead of pretrained=False
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.model.fc.in_features, 2)
        )

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()

        self.mc_dropout_iterations = mc_dropout_iterations

        def enable_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()
        self.enable_dropout = enable_dropout

        # Dynamically access the target layer
        layer_dict = {
            "layer4[-1]": self.model.layer4[-1],
            "layer4[-2]": self.model.layer4[-2],
            "layer3[-1]": self.model.layer3[-1]
        }
        # Pass the parameters to EnhancedGradCAM
        self.grad_cam = EnhancedGradCAM(
            self.model, 
            layer_dict[target_layer],
            threshold=threshold,
            max_area_fraction=max_area_fraction,
            min_area_fraction=min_area_fraction,
            use_morph_ops=use_morph_ops,
            use_gradcam_plus_plus=use_gradcam_plus_plus
        )

        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.classes = ['Normal', 'Pneumonia']
        self.input_size = input_size

    def forward(self, x):
        return self.model(x)

    def predict(self, image):
        original_width, original_height = image.size
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