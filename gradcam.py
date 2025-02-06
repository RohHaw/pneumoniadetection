from gradcam_base import GradCAM
import torch
import cv2
import numpy as np

class EnhancedGradCAM(GradCAM):
    def __init__(self, model, target_layer):
        super().__init__(model, target_layer)
        
    def generate_with_boxes(self, input_image, threshold=0.5):
        # Generate basic heatmap
        heatmap = self.generate(input_image)
        
        # Convert heatmap to numpy array
        heatmap_np = np.uint8(255 * heatmap)
        
        # Find contours in the heatmap
        binary = cv2.threshold(heatmap_np, int(255 * threshold), 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get original image dimensions
        height, width = heatmap_np.shape
        
        # Draw bounding boxes
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Scale coordinates to original image size
            boxes.append({
                'x': int(x * (input_image.size[0] / width)),
                'y': int(y * (input_image.size[1] / height)),
                'width': int(w * (input_image.size[0] / width)),
                'height': int(h * (input_image.size[1] / height))
            })
        
        return heatmap, boxes

    def get_region_descriptions(self, boxes):
        descriptions = []
        for box in boxes:
            # Calculate position in image (top/bottom, left/right)
            x_center = box['x'] + box['width']/2
            y_center = box['y'] + box['height']/2
            
            position = []
            if y_center < self.image_height/3:
                position.append("upper")
            elif y_center < 2*self.image_height/3:
                position.append("middle")
            else:
                position.append("lower")
                
            if x_center < self.image_width/3:
                position.append("left")
            elif x_center < 2*self.image_width/3:
                position.append("central")
            else:
                position.append("right")
                
            descriptions.append(f"{' '.join(position)} region")
        
        return descriptions