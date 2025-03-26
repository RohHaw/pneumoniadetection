import os
import pandas as pd
from PIL import Image
import numpy as np
import pydicom
from classifier import PneumoniaClassifier
import json

# Paths to RSNA dataset
data_dir = "D:/rsna_data/train_images"  # Adjust if your .dcm files are elsewhere
csv_path = "D:/rsna_data/labels/stage_2_train_labels.csv"
model_path = "Training/best_rsna_model.pth"

# Verify directory exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory not found: {data_dir}. Please check the path.")
print(f"Data directory: {data_dir}")
print(f"First 5 files in directory: {os.listdir(data_dir)[:5]}")

# Load a subset of RSNA data for testing
print(f"Loading CSV from: {csv_path}")
df = pd.read_csv(csv_path)
print(f"CSV loaded. Total pneumonia cases: {len(df[df['Target'] == 1].dropna())}")

pneumonia_df = df[df['Target'] == 1].dropna().sample(200, random_state=42)  # 200 images with bounding boxes
print(f"Selected {len(pneumonia_df)} pneumonia cases. Sample patientIds:\n{pneumonia_df['patientId'].tolist()[:5]}")

# Construct image paths and ground truth boxes
image_paths = [os.path.join(data_dir, f"{pid}.dcm") for pid in pneumonia_df['patientId']]
ground_truth_boxes = [
    [{'x': row['x'], 'y': row['y'], 'width': row['width'], 'height': row['height']}]
    for _, row in pneumonia_df.iterrows()
]

# Function to load and convert DICOM to PIL Image
def load_dicom_as_pil(dicom_path):
    """Convert a DICOM file to a PIL Image in RGB format."""
    print(f"Attempting to load: {dicom_path}")
    try:
        if not os.path.exists(dicom_path):
            print(f"File does not exist: {dicom_path}")
            return None
        dicom = pydicom.dcmread(dicom_path)
        image = dicom.pixel_array
        # Normalize pixel values to 0-255
        image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
        image = Image.fromarray(image.astype(np.uint8)).convert('RGB')
        print(f"Successfully loaded: {dicom_path}")
        return image
    except Exception as e:
        print(f"Error loading {dicom_path}: {str(e)}")
        return None

# Load test images with error handling
test_images = []
valid_ground_truth_boxes = []
skipped_images = []
for path, gt_boxes in zip(image_paths, ground_truth_boxes):
    img = load_dicom_as_pil(path)
    if img is not None:
        test_images.append(img)
        valid_ground_truth_boxes.append(gt_boxes)
    else:
        skipped_images.append(path)

if not test_images:
    print(f"Directory contents (first 5 files): {os.listdir(data_dir)[:5]}")
    raise ValueError("No valid images loaded. Check if patientIds match file names in the directory.")
if skipped_images:
    print(f"Skipped {len(skipped_images)} images due to errors: {skipped_images}")

def compute_iou(pred_box, gt_box):
    """Compute IoU between a predicted box and a ground truth box."""
    x1, y1, w1, h1 = pred_box['x'], pred_box['y'], pred_box['width'], pred_box['height']
    x2, y2, w2, h2 = gt_box['x'], gt_box['y'], gt_box['width'], gt_box['height']
    
    inter_xmin = max(x1, x2)
    inter_ymin = max(y1, y2)
    inter_xmax = min(x1 + w1, x2 + w2)
    inter_ymax = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    union_area = w1 * h1 + w2 * h2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def evaluate_iou(classifier, images, gt_boxes_list):
    """Evaluate IoU for a set of images and ground truth boxes."""
    iou_scores = []
    for img, gt_boxes in zip(images, gt_boxes_list):
        results = classifier.predict(img)
        pred_boxes = results.get('boxes', [])
        
        if not pred_boxes:  # No boxes detected
            iou_scores.append(0)
            continue
        
        # Find the best IoU for each ground truth box
        max_iou = 0
        for gt_box in gt_boxes:
            ious = [compute_iou(pred_box, gt_box) for pred_box in pred_boxes]
            max_iou = max(max_iou, max(ious) if ious else 0)
        iou_scores.append(max_iou)
    
    return np.mean(iou_scores), iou_scores

# Use the best hyperparameters
best_threshold = 0.4
best_max_area = 0.6
best_min_area = 0.005

print(f"Evaluating with best hyperparameters: threshold={best_threshold}, max_area={best_max_area}, min_area={best_min_area}")

# Initialize classifier with best hyperparameters
classifier = PneumoniaClassifier(
    model_path=model_path,
    threshold=best_threshold,
    max_area_fraction=best_max_area,
    min_area_fraction=best_min_area
)

# Evaluate IoU
avg_iou, iou_scores = evaluate_iou(classifier, test_images, valid_ground_truth_boxes)

# Log the result
result = {
    'threshold': best_threshold,
    'max_area_fraction': best_max_area,
    'min_area_fraction': best_min_area,
    'avg_iou': avg_iou,
    'individual_iou_scores': iou_scores
}

# Save results to a file
output_file = 'gradcam_best_results_200_images.json'
with open(output_file, 'w') as f:
    json.dump(result, f, indent=4)
print(f"Results saved to {output_file}")

print(f"\nAverage IoU over 200 images: {avg_iou:.3f}")
print(f"Individual IoU scores (first 5): {iou_scores[:5]}")