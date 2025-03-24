import os
import pandas as pd
import pydicom
from PIL import Image
import numpy as np
import cv2
from classifier import PneumoniaClassifier


RSNA_DATA_DIR = "/vol/scratch/SoC/misc/2024/sc21r2h/rsna_data"  # Directory containing RSNA dataset
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))  # Current project directory
MODEL_PATH = os.path.join(PROJECT_DIR, "best_model_final.pth")  # Path to trained model weights


def compute_iou(box1: dict, box2: dict) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (dict): First box with keys 'x', 'y', 'width', 'height'
        box2 (dict): Second box with keys 'x', 'y', 'width', 'height'

    Returns:
        float: IoU value between 0 and 1, or 0 if no union exists
    """
    # Extract coordinates for both boxes
    x1_min, y1_min = box1['x'], box1['y']
    x1_max, y1_max = x1_min + box1['width'], y1_min + box1['height']
    x2_min, y2_min = box2['x'], box2['y']
    x2_max, y2_max = x2_min + box2['width'], y2_min + box2['height']

    # Calculate intersection coordinates
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Compute intersection and union areas
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = box1['width'] * box1['height']
    box2_area = box2['width'] * box2['height']
    union_area = box1_area + box2_area - inter_area

    # Return IoU, handling division by zero
    return inter_area / union_area if union_area > 0 else 0


def evaluate_gradcam(classifier: PneumoniaClassifier, image_dir: str, labels_file: str, 
                    sample_size: int = None, debug_limit: int = 5) -> None:
    """Evaluate GradCAM performance against ground truth bounding boxes.

    This function assesses the classifier's localisation accuracy by comparing
    GradCAM-generated boxes with ground truth annotations using IoU metrics.

    Args:
        classifier (PneumoniaClassifier): Trained pneumonia classifier instance
        image_dir (str): Directory containing DICOM image files
        labels_file (str): Path to CSV file with ground truth labels
        sample_size (int, optional): Number of samples to evaluate; if None, use all
        debug_limit (int): Maximum number of debug images to save
    """
    # Load and filter labels for pneumonia cases
    labels = pd.read_csv(labels_file)
    pneumonia_cases = labels[labels["Target"] == 1].drop_duplicates(subset="patientId")
    
    # Apply sampling if specified
    if sample_size:
        pneumonia_cases = pneumonia_cases.sample(n=sample_size, random_state=42)
    
    # Initialise lists and counters
    iou_scores = []
    processed_count = 0
    debug_count = 0
    
    print(f"Evaluating {len(pneumonia_cases)} pneumonia cases...")
    
    # Process each pneumonia case
    for _, row in pneumonia_cases.iterrows():
        patient_id = row["patientId"]
        image_path = os.path.join(image_dir, f"{patient_id}.dcm")
        
        # Skip if image file is missing
        if not os.path.exists(image_path):
            print(f"Skipping {patient_id}: Image not found")
            continue
        
        try:
            # Read and convert DICOM image to RGB
            ds = pydicom.dcmread(image_path)
            image = Image.fromarray(ds.pixel_array).convert("RGB")
            
            # Get classifier predictions
            results = classifier.predict(image)
            gradcam_boxes = results["boxes"]
            heatmap = results["gradcam"]  # Heatmap overlay from classifier
            
            # Extract ground truth boxes
            gt_boxes_df = labels[(labels["patientId"] == patient_id) & (labels["Target"] == 1)]
            gt_boxes = [
                {'x': row['x'], 'y': row['y'], 'width': row['width'], 'height': row['height']}
                for _, row in gt_boxes_df.iterrows()
            ]
            
            # Skip if no ground truth boxes available
            if not gt_boxes:
                print(f"Skipping {patient_id}: No ground truth boxes")
                continue
            
            # Generate debug visualisations if within limit
            if debug_count < debug_limit:
                img_array = np.array(image)
                # Draw GradCAM boxes in green
                for box in gradcam_boxes:
                    x, y, w, h = box['x'], box['y'], box['width'], box['height']
                    cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Draw ground truth boxes in red
                for gt_box in gt_boxes:
                    x, y, w, h = int(gt_box['x']), int(gt_box['y']), int(gt_box['width']), int(gt_box['height'])
                    cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # Save debug images
                cv2.imwrite(os.path.join(PROJECT_DIR, f"debug_boxes_{patient_id}.jpg"), img_array)
                cv2.imwrite(os.path.join(PROJECT_DIR, f"debug_heatmap_{patient_id}.jpg"), np.array(heatmap))
                debug_count += 1
            
            # Display debugging information
            print(f"{patient_id}:")
            print(f"  Class = {results['class']}, Pneumonia Prob = {results['probabilities']['Pneumonia']:.2f}%")
            print(f"  GradCAM Boxes = {gradcam_boxes}")
            print(f"  Ground Truth Boxes = {gt_boxes}")
            
            # Calculate maximum IoU for each GradCAM box
            max_iou_per_gbox = []
            for gbox in gradcam_boxes:
                ious = [compute_iou(gbox, gt_box) for gt_box in gt_boxes]
                max_iou_per_gbox.append(max(ious) if ious else 0)
            
            # Compute and store average IoU if boxes exist
            if max_iou_per_gbox:
                avg_iou = np.mean(max_iou_per_gbox)
                iou_scores.append(avg_iou)
                print(f"  Average IoU = {avg_iou:.4f}")
            else:
                print(f"  No GradCAM boxes generated")
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {patient_id}: {str(e)}")
            continue
    
    # Summarise evaluation results
    if iou_scores:
        mean_iou = np.mean(iou_scores)
        std_iou = np.std(iou_scores)
        print(f"\nEvaluation Complete ({processed_count} images processed):")
        print(f"Mean IoU: {mean_iou:.4f}")
        print(f"Standard Deviation IoU: {std_iou:.4f}")
        print(f"Minimum IoU: {min(iou_scores):.4f}")
        print(f"Maximum IoU: {max(iou_scores):.4f}")
    else:
        print("No valid IoU scores calculated.")


def main() -> None:
    """Main function to run GradCAM evaluation."""
    # Define paths to images and labels
    image_dir = os.path.join(RSNA_DATA_DIR, "train_images")
    labels_file = os.path.join(RSNA_DATA_DIR, "labels", "stage_2_train_labels.csv")
    
    # Check if required files and directories exist
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    # Initialise classifier and run evaluation
    classifier = PneumoniaClassifier(model_path=MODEL_PATH)
    evaluate_gradcam(classifier, image_dir, labels_file, sample_size=100, debug_limit=5)


if __name__ == "__main__":
    main()