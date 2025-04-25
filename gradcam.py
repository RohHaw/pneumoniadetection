"""
Module defining the EnhancedGradCAM class for visual explanations in pneumonia detection.

This module implements an enhanced version of Gradient-weighted Class Activation Mapping (Grad-CAM)
to generate heatmaps and bounding boxes for regions of interest in chest X-ray images. It supports
morphological operations for heatmap refinement, bounding box merging based on Intersection over
Union (IoU), and region descriptions for interpretability.

Author: Rohman Hawrylak
Date: April 2025
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2

class EnhancedGradCAM:
    """
    Enhanced Grad-CAM implementation for generating visual explanations.

    This class generates heatmaps highlighting regions of a chest X-ray image that influence a
    model's classification decision. It extends Grad-CAM by adding bounding box extraction,
    morphological operations for heatmap refinement, and IoU-based box merging. Region
    descriptions are provided for interpretability.

    Attributes:
        model (torch.nn.Module): The neural network model for which Grad-CAM is computed.
        target_layer (torch.nn.Module): The target layer for feature extraction.
        threshold (float): Threshold for binarising the heatmap.
        max_area_fraction (float): Maximum allowed area fraction for bounding boxes.
        min_area_fraction (float): Minimum allowed area fraction for bounding boxes.
        use_morph_ops (bool): Whether to apply morphological operations to the heatmap.
        image_width (int): Width of the input image.
        image_height (int): Height of the input image.
        gradients (torch.Tensor): Stored gradients from the target layer.
        features (torch.Tensor): Stored features from the target layer.
        hooks (list): List of registered forward and backward hooks.
    """
    def __init__(self, model, target_layer, threshold=0.4, max_area_fraction=0.5, 
                 min_area_fraction=0.05, use_morph_ops=True):
        """
        Initialise the EnhancedGradCAM instance.

        Args:
            model (torch.nn.Module): The neural network model (e.g., ResNet-50).
            target_layer (torch.nn.Module): The layer for Grad-CAM feature extraction.
            threshold (float, optional): Threshold for binarising the heatmap. Defaults to 0.4.
            max_area_fraction (float, optional): Maximum box area fraction. Defaults to 0.5.
            min_area_fraction (float, optional): Minimum box area fraction. Defaults to 0.05.
            use_morph_ops (bool, optional): Apply morphological operations. Defaults to True.
        """
        self.model = model
        self.target_layer = target_layer
        self.threshold = threshold
        self.max_area_fraction = max_area_fraction
        self.min_area_fraction = min_area_fraction
        self.use_morph_ops = use_morph_ops
        self.image_width = None
        self.image_height = None
        self.gradients = None
        self.features = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """
        Register forward and backward hooks to capture features and gradients.

        Hooks are attached to the target layer to store features during the forward pass and
        gradients during the backward pass.
        """
        def save_gradient(grad):
            self.gradients = grad.detach()
            
        def save_features(module, input, output):
            self.features = output.detach()
        
        handle_forward = self.target_layer.register_forward_hook(save_features)
        handle_backward = self.target_layer.register_full_backward_hook(
            lambda module, grad_input, grad_output: save_gradient(grad_output[0])
        )
        self.hooks.extend([handle_forward, handle_backward])

    def _release_hooks(self):
        """
        Remove all registered hooks to free resources.

        Clears the hooks list after removal.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def generate(self, input_tensor):
        """
        Generate a Grad-CAM heatmap for the input tensor.

        Computes the heatmap by combining feature maps and gradients from the target layer,
        applying ReLU and normalising the result.

        Args:
            input_tensor (torch.Tensor): Input image tensor.

        Returns:
            numpy.ndarray: Normalised Grad-CAM heatmap.

        Raises:
            ValueError: If gradients or features are not captured.
        """
        if isinstance(input_tensor, torch.Tensor):
            self.image_height = input_tensor.size(-2)
            self.image_width = input_tensor.size(-1)
        
        # Clear previous gradients
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output.max()
        score.backward()
        
        gradients = self.gradients
        features = self.features
        
        if gradients is None or features is None:
            raise ValueError("Gradients or features are None.")
        
        # Compute weights as the mean of gradients over spatial dimensions
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * features, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Resize heatmap to match input dimensions
        cam = F.interpolate(cam, size=(self.image_height, self.image_width), mode='bilinear', align_corners=False)
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        
        return cam.squeeze().cpu().numpy()

    def generate_with_boxes(self, input_image, threshold=None, max_area_fraction=None, 
                           min_area_fraction=None, original_width=None, original_height=None):
        """
        Generate a Grad-CAM heatmap and extract bounding boxes.

        Processes the heatmap to identify regions of interest, applies morphological operations
        (if enabled), and extracts bounding boxes based on contour detection. Boxes are filtered
        by area constraints and scaled to the original image dimensions.

        Args:
            input_image (torch.Tensor): Input image tensor.
            threshold (float, optional): Binarisation threshold. Defaults to self.threshold.
            max_area_fraction (float, optional): Maximum box area fraction. Defaults to self.max_area_fraction.
            min_area_fraction (float, optional): Minimum box area fraction. Defaults to self.min_area_fraction.
            original_width (int, optional): Original image width for scaling.
            original_height (int, optional): Original image height for scaling.

        Returns:
            tuple: (heatmap, boxes)
                - heatmap (numpy.ndarray): Normalised Grad-CAM heatmap.
                - boxes (list): List of dictionaries containing bounding box coordinates.
        """
        heatmap = self.generate(input_image)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Convert heatmap to 8-bit for contour detection
        heatmap_np = np.uint8(255 * heatmap)
        thresh = threshold if threshold is not None else self.threshold
        binary = cv2.threshold(heatmap_np, int(255 * thresh), 255, cv2.THRESH_BINARY)[1]
        
        # Apply morphological operations to refine the binary mask
        if self.use_morph_ops:
            kernel = np.ones((5, 5), np.uint8)
            binary = cv2.erode(binary, kernel, iterations=1)
            binary = cv2.dilate(binary, kernel, iterations=1)
        
        # Detect contours in the binary mask
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        height, width = heatmap_np.shape
        image_area = height * width
        
        # Calculate scaling factors for original dimensions
        scale_width = original_width / width if original_width else 1
        scale_height = original_height / height if original_height else 1
        
        max_area_frac = max_area_fraction if max_area_fraction is not None else self.max_area_fraction
        min_area_frac = min_area_fraction if min_area_fraction is not None else self.min_area_fraction
        
        # Extract and filter bounding boxes
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            box_area = w * h
            if (box_area / image_area <= max_area_frac and 
                box_area / image_area >= min_area_frac and 
                w > 5 and h > 5):
                box = {
                    'x': int(x * scale_width),
                    'y': int(y * scale_height),
                    'width': int(w * scale_width),
                    'height': int(h * scale_height)
                }
                # Ensure box coordinates are within image bounds
                box['x'] = max(0, min(box['x'], original_width - 1))
                box['y'] = max(0, min(box['y'], original_height - 1))
                box['width'] = min(box['width'], original_width - box['x'])
                box['height'] = min(box['height'], original_height - box['y'])
                boxes.append(box)
        
        # Merge overlapping boxes based on IoU
        merged_boxes = []
        while boxes:
            box = boxes.pop(0)
            i = 0
            while i < len(boxes):
                other = boxes[i]
                x1_min, y1_min = box['x'], box['y']
                x1_max, y1_max = x1_min + box['width'], y1_min + box['height']
                x2_min, y2_min = other['x'], other['y']
                x2_max, y2_max = x2_min + other['width'], y2_min + other['height']
                
                # Calculate IoU
                inter_x_min = max(x1_min, x2_min)
                inter_y_min = max(y1_min, y2_min)
                inter_x_max = min(x1_max, x2_max)
                inter_y_max = min(y1_max, y2_max)
                
                inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
                box1_area = box['width'] * box['height']
                box2_area = other['width'] * other['height']
                union_area = box1_area + box2_area - inter_area
                
                iou = inter_area / union_area if union_area > 0 else 0
                
                # Merge boxes with IoU > 0.3
                if iou > 0.3:
                    merged_box = {
                        'x': min(box['x'], other['x']),
                        'y': min(box['y'], other['y']),
                        'width': max(box['x'] + box['width'], other['x'] + other['width']) - min(box['x'], other['x']),
                        'height': max(box['y'] + box['height'], other['y'] + other['height']) - min(box['y'], other['y'])
                    }
                    box = merged_box
                    boxes.pop(i)
                else:
                    i += 1
            merged_boxes.append(box)
        
        return heatmap, merged_boxes

    def get_region_descriptions(self, boxes):
        """
        Generate textual descriptions of bounding box locations.

        Describes the approximate location of each bounding box (e.g., 'Upper left region')
        based on its centre point relative to the image dimensions.

        Args:
            boxes (list): List of dictionaries containing bounding box coordinates.

        Returns:
            list: List of strings describing the location of each box.
        """
        descriptions = []
        for box in boxes:
            x_center = box['x'] + box['width']/2
            y_center = box['y'] + box['height']/2
            
            # Determine vertical position
            position = []
            if y_center < self.image_height/3:
                position.append("Upper")
            elif y_center < 2*self.image_height/3:
                position.append("Middle")
            else:
                position.append("Lower")
                
            # Determine horizontal position
            if x_center < self.image_width/3:
                position.append("left")
            elif x_center < 2*self.image_width/3:
                position.append("central")
            else:
                position.append("right")
                
            descriptions.append(f"{' '.join(position)} region")
        
        return descriptions

    def __del__(self):
        """
        Destructor to ensure hooks are released when the object is deleted.
        """
        self._release_hooks()