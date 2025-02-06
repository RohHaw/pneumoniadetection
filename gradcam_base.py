import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None
        self.image_width = None
        self.image_height = None
        
        # Register hooks
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        def save_gradient(grad):
            self.gradients = grad.detach()
            
        def save_features(module, input, output):
            self.features = output.detach()
        
        # Register forward and backward hooks
        handle_forward = self.target_layer.register_forward_hook(save_features)
        handle_backward = self.target_layer.register_full_backward_hook(
            lambda module, grad_input, grad_output: save_gradient(grad_output[0])
        )
        
        self.hooks.extend([handle_forward, handle_backward])
        
    def _release_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
            
    def generate(self, input_tensor):
        # Store image dimensions
        if isinstance(input_tensor, torch.Tensor):
            self.image_height = input_tensor.size(-2)
            self.image_width = input_tensor.size(-1)
        
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Get the score for the predicted class
        score = output.max()
        
        # Backward pass
        score.backward()
        
        # Get gradients and features
        gradients = self.gradients
        features = self.features
        
        # Calculate weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Generate cam
        cam = torch.sum(weights * features, dim=1, keepdim=True)
        cam = F.relu(cam)  # Apply ReLU to focus on positive contributions
        
        # Normalize the cam
        cam = F.interpolate(cam, size=(self.image_height, self.image_width), mode='bilinear', align_corners=False)
        cam -= cam.min()
        cam /= cam.max()
        
        # Convert to numpy array
        cam = cam.squeeze().cpu().numpy()
        
        return cam
        
    def __del__(self):
        self._release_hooks()