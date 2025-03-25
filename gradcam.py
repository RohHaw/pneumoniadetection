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
        
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
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
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
            
    def generate(self, input_tensor):
        if isinstance(input_tensor, torch.Tensor):
            self.image_height = input_tensor.size(-2)
            self.image_width = input_tensor.size(-1)
        
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        score = output.max()
        score.backward()
        
        gradients = self.gradients
        features = self.features
        
        if gradients is None or features is None:
            raise ValueError("Gradients or features are None. Ensure hooks are properly set up and the backward pass completed.")
        
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * features, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = F.interpolate(cam, size=(self.image_height, self.image_width), mode='bilinear', align_corners=False)
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        
        return cam.squeeze().cpu().numpy()
        
    def __del__(self):
        self._release_hooks()

class GradCAMPlusPlus(GradCAM):
    def generate(self, input_tensor):
        if isinstance(input_tensor, torch.Tensor):
            self.image_height = input_tensor.size(-2)
            self.image_width = input_tensor.size(-1)
        
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        score = output.max()
        score.backward()
        
        gradients = self.gradients  # Shape: (batch, channels, height, width)
        features = self.features    # Shape: (batch, channels, height, width)
        
        if gradients is None or features is None:
            raise ValueError("Gradients or features are None. Ensure hooks are properly set up and the backward pass completed.")
        
        # Compute GradCAM++ weights
        alpha_num = gradients ** 2
        alpha_denom = 2 * gradients ** 2 + torch.sum(features * gradients ** 3, dim=(2, 3), keepdim=True)
        alpha = alpha_num / (alpha_denom + 1e-8)  # Shape: (batch, channels, height, width)
        
        weights = torch.sum(F.relu(gradients) * alpha, dim=(2, 3), keepdim=True)  # Shape: (batch, channels, 1, 1)
        cam = torch.sum(weights * features, dim=1, keepdim=True)  # Shape: (batch, 1, height, width)
        cam = F.relu(cam)
        
        cam = F.interpolate(cam, size=(self.image_height, self.image_width), mode='bilinear', align_corners=False)
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        
        return cam.squeeze().cpu().numpy()

class EnhancedGradCAM:
    def __init__(self, model, target_layer, threshold=0.4, max_area_fraction=0.5, 
                 min_area_fraction=0.05, use_morph_ops=True, use_gradcam_plus_plus=False):
        if use_gradcam_plus_plus:
            self.gradcam = GradCAMPlusPlus(model, target_layer)
        else:
            self.gradcam = GradCAM(model, target_layer)
        self.threshold = threshold
        self.max_area_fraction = max_area_fraction
        self.min_area_fraction = min_area_fraction
        self.use_morph_ops = use_morph_ops
        self.image_width = None
        self.image_height = None
        
    def generate(self, input_tensor):
        if isinstance(input_tensor, torch.Tensor):
            self.image_height = input_tensor.size(-2)
            self.image_width = input_tensor.size(-1)
        return self.gradcam.generate(input_tensor)

    def generate_with_boxes(self, input_image, original_width=None, original_height=None):
        heatmap = self.generate(input_image)
        
        print(f"Heatmap min: {heatmap.min():.4f}, max: {heatmap.max():.4f}, mean: {heatmap.mean():.4f}")
        
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        heatmap_np = np.uint8(255 * heatmap)
        binary = cv2.threshold(heatmap_np, int(255 * self.threshold), 255, cv2.THRESH_BINARY)[1]
        
        if self.use_morph_ops:
            kernel = np.ones((5, 5), np.uint8)
            binary = cv2.erode(binary, kernel, iterations=1)
            binary = cv2.dilate(binary, kernel, iterations=1)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        height, width = heatmap_np.shape
        image_area = height * width
        
        scale_width = original_width / width if original_width else 1
        scale_height = original_height / height if original_height else 1
        
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            box_area = w * h
            
            if (box_area / image_area <= self.max_area_fraction and 
                box_area / image_area >= self.min_area_fraction and 
                w > 5 and h > 5):
                box = {
                    'x': int(x * scale_width),
                    'y': int(y * scale_height),
                    'width': int(w * scale_width),
                    'height': int(h * scale_height)
                }
                box['x'] = max(0, min(box['x'], original_width - 1))
                box['y'] = max(0, min(box['y'], original_height - 1))
                box['width'] = min(box['width'], original_width - box['x'])
                box['height'] = min(box['height'], original_height - box['y'])
                boxes.append(box)
        
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
                
                inter_x_min = max(x1_min, x2_min)
                inter_y_min = max(y1_min, y2_min)
                inter_x_max = min(x1_max, x2_max)
                inter_y_max = min(y1_max, y2_max)
                
                inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
                box1_area = box['width'] * box['height']
                box2_area = other['width'] * other['height']
                union_area = box1_area + box2_area - inter_area
                
                iou = inter_area / union_area if union_area > 0 else 0
                
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
        
        print(f"Number of boxes after merging: {len(merged_boxes)}")
        return heatmap, merged_boxes

    def get_region_descriptions(self, boxes):
        descriptions = []
        for box in boxes:
            x_center = box['x'] + box['width']/2
            y_center = box['y'] + box['height']/2
            
            position = []
            if y_center < self.image_height/3:
                position.append("Upper")
            elif y_center < 2*self.image_height/3:
                position.append("Middle")
            else:
                position.append("Lower")
                
            if x_center < self.image_width/3:
                position.append("left")
            elif x_center < 2*self.image_width/3:
                position.append("central")
            else:
                position.append("right")
                
            descriptions.append(f"{' '.join(position)} region")
        
        return descriptions