import os
import torch
from PIL import Image
import pydicom
import pandas as pd
import numpy as np
import cv2
from classifier import PneumoniaClassifier  # Your custom classifier

# Paths
data_dir = "/vol/scratch/SoC/misc/2024/sc21r2h/rsna_data/train_images"
csv_path = "/vol/scratch/SoC/misc/2024/sc21r2h/rsna_data/labels/stage_2_train_labels.csv"
model_path = "Training/UCSD/model_RSNA.pth"
output_dir = "evaluation_images_and_data/RSNA_Model/gradcam_eval"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the RSNA labels
labels_df = pd.read_csv(csv_path)

# Initialize the classifier
classifier = PneumoniaClassifier(model_path=model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier.to(device)

def load_dicom_image(file_path):
    """Load a DICOM image and convert it to a PIL Image."""
    dicom = pydicom.dcmread(file_path)
    image = dicom.pixel_array
    # Normalize to 0-255 range
    image = (image - image.min()) / (image.max() - image.min()) * 255
    image = image.astype(np.uint8)
    # Convert to RGB
    image = np.stack([image] * 3, axis=-1)
    return Image.fromarray(image)

def get_ground_truth_boxes(patient_id, labels_df):
    """Extract ground truth bounding boxes from the RSNA labels CSV."""
    patient_data = labels_df[labels_df['patientId'] == patient_id]
    boxes = []
    for _, row in patient_data.iterrows():
        if row['Target'] == 1:  # Positive pneumonia case
            boxes.append({
                'x': int(row['x']),
                'y': int(row['y']),
                'width': int(row['width']),
                'height': int(row['height'])
            })
    return boxes

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes."""
    x1_min = box1['x']
    y1_min = box1['y']
    x1_max = x1_min + box1['width']
    y1_max = y1_min + box1['height']

    x2_min = box2['x']
    y2_min = box2['y']
    x2_max = x2_min + box2['width']
    y2_max = y2_min + box2['height']

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = box1['width'] * box1['height']
    box2_area = box2['width'] * box2['height']
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def save_visualization(image, pred_boxes, gt_boxes, patient_id, output_dir):
    """Save an image with predicted and ground truth boxes overlaid."""
    image_np = np.array(image)
    # Draw predicted boxes (green)
    for box in pred_boxes:
        x, y, w, h = box['x'], box['y'], box['width'], box['height']
        cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for predicted
    # Draw ground truth boxes (red)
    for box in gt_boxes:
        x, y, w, h = box['x'], box['y'], box['width'], box['height']
        cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for ground truth
    
    # Save the image
    output_path = os.path.join(output_dir, f"{patient_id}_gradcam_eval.png")
    cv2.imwrite(output_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    return output_path

def test_gradcam_iou(num_samples=200, num_images_to_save=20):
    """Test IoU of GradCAM bounding boxes, save raw data, and select 20 images."""
    iou_scores = []
    processed_samples = 0
    data_records = []  # For CSV output

    print(f"Processing {num_samples} samples...")

    # Iterate over the dataset
    for idx, row in labels_df.iterrows():
        if processed_samples >= num_samples:
            break

        patient_id = row['patientId']
        dicom_path = os.path.join(data_dir, f"{patient_id}.dcm")

        if not os.path.exists(dicom_path):
            print(f"Skipping {patient_id}: DICOM file not found")
            continue

        # Load and preprocess the image
        try:
            image = load_dicom_image(dicom_path)
        except Exception as e:
            print(f"Error loading {patient_id}: {str(e)}")
            continue

        # Get ground truth boxes (only for pneumonia cases)
        gt_boxes = get_ground_truth_boxes(patient_id, labels_df)
        if not gt_boxes:  # Skip if no pneumonia (no ground truth boxes)
            continue

        # Predict with the classifier
        try:
            results = classifier.predict(image)
            pred_boxes = results['boxes']  # Predicted bounding boxes from GradCAM
        except Exception as e:
            print(f"Error predicting {patient_id}: {str(e)}")
            continue

        # Calculate IoU for each predicted box against ground truth
        sample_ious = []
        for pred_box in pred_boxes:
            max_iou = 0
            for gt_box in gt_boxes:
                iou = calculate_iou(pred_box, gt_box)
                max_iou = max(max_iou, iou)
            sample_ious.append(max_iou)

        # Average IoU for this sample (if there are predicted boxes)
        if sample_ious:
            avg_iou = sum(sample_ious) / len(sample_ious)
            iou_scores.append(avg_iou)
        else:
            avg_iou = 0  # No predicted boxes
            print(f"Patient {patient_id} - No predicted boxes")

        # Record data for CSV
        data_records.append({
            'PatientID': patient_id,
            'PredictedBoxes': len(pred_boxes),
            'GroundTruthBoxes': len(gt_boxes),
            'AverageIoU': avg_iou,
            'PneumoniaProbability': results['probabilities']['Pneumonia']  # Added for more context
        })

        processed_samples += 1
        if processed_samples % 50 == 0:  # Progress update every 50 samples
            print(f"Processed {processed_samples}/{num_samples} samples")

    # Summary
    if iou_scores:
        overall_avg_iou = sum(iou_scores) / len(iou_scores)
        print(f"\nProcessed {len(iou_scores)} valid samples with pneumonia.")
        print(f"Overall Average IoU: {overall_avg_iou:.4f}")
    else:
        print("No valid IoU scores calculated.")

    # Save raw data to CSV
    df = pd.DataFrame(data_records)
    csv_path = os.path.join(output_dir, "gradcam_iou_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Raw data saved to {csv_path}")

    # Select 20 representative images based on IoU distribution
    if len(data_records) > num_images_to_save:
        # Sort by IoU to get a range of values
        df_sorted = df.sort_values(by='AverageIoU')
        # Select evenly spaced indices
        indices = np.linspace(0, len(df_sorted) - 1, num_images_to_save, dtype=int)
        selected_records = df_sorted.iloc[indices]
        
        print(f"\nSaving {num_images_to_save} select visualizations...")
        for _, record in selected_records.iterrows():
            patient_id = record['PatientID']
            dicom_path = os.path.join(data_dir, f"{patient_id}.dcm")
            try:
                image = load_dicom_image(dicom_path)
                results = classifier.predict(image)
                pred_boxes = results['boxes']
                gt_boxes = get_ground_truth_boxes(patient_id, labels_df)
                vis_path = save_visualization(image, pred_boxes, gt_boxes, patient_id, output_dir)
                print(f"Saved visualization for {patient_id} (IoU: {record['AverageIoU']:.4f}) at {vis_path}")
            except Exception as e:
                print(f"Error saving visualization for {patient_id}: {str(e)}")

if __name__ == "__main__":
    test_gradcam_iou(num_samples=200, num_images_to_save=20)  # Test on 200 samples, save 20 images