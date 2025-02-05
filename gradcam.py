import torch
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_features)
        self.target_layer.register_full_backward_hook(self.save_gradients)
    
    def save_features(self, module, input, output):
        self.features = output.detach()
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_image):
        # Get the most likely prediction
        model_output = self.model(input_image)
        pred_class = model_output.argmax(dim=1)
        
        # Zero all existing gradients
        self.model.zero_grad()
        
        # Get gradients of the target class
        model_output[:, pred_class].backward()
        
        # Calculate GradCAM
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.features.shape[1]):
            self.features[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(self.features, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap /= torch.max(heatmap)
        
        return heatmap.cpu().numpy()