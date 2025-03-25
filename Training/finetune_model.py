import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from classifier import PneumoniaClassifier
from dataset import RSNADataset, transform  # Assuming dataset.py contains RSNADataset and transform
import torchvision.transforms as transforms  # Add this import

# Define paths
RSNA_DATA_DIR = "D:/rsna_data"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "Training/best_model_split_vis.pth")
FINETUNED_MODEL_PATH = os.path.join(PROJECT_DIR, "Training/finetuned_model.pth")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_iou(box1: dict, box2: dict) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1_min, y1_min = box1['x'], box1['y']
    x1_max, y1_max = x1_min + box1['width'], y1_min + box1['height']
    x2_min, y2_min = box2['x'], box2['y']
    x2_max, y2_max = x2_min + box2['width'], y2_min + box2['height']

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = box1['width'] * box1['height']
    box2_area = box2['width'] * box2['height']
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def evaluate_classification(model, dataloader):
    """Evaluate classification performance on a dataloader."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in dataloader:  # Unpack 3 values, ignore patient_id
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    # Debug: Print label and prediction distribution
    print(f"Label distribution: {np.bincount(all_labels)} (0=Normal, 1=Pneumonia)")
    print(f"Prediction distribution: {np.bincount(all_preds)} (0=Normal, 1=Pneumonia)")
    
    return accuracy, precision, recall, f1

def evaluate_localization(model, dataloader, labels_df):
    """Evaluate localization performance (IoU) on a dataloader."""
    model.eval()
    iou_scores = []
    
    for images, _, patient_ids in dataloader:  # Unpack 3 values, use patient_ids
        for img, pid in zip(images, patient_ids):
            img = img.unsqueeze(0).to(device)
            # Convert tensor back to PIL for prediction
            img_pil = transforms.ToPILImage()(img.squeeze(0).cpu())  # Use the imported transforms module
            
            results = model.predict(img_pil)
            gradcam_boxes = results["boxes"]
            
            # Extract ground truth boxes
            gt_boxes_df = labels_df[(labels_df["patientId"] == pid) & (labels_df["Target"] == 1)]
            gt_boxes = [
                {'x': row['x'], 'y': row['y'], 'width': row['width'], 'height': row['height']}
                for _, row in gt_boxes_df.iterrows()
            ]
            
            if not gt_boxes:
                continue
            
            # Calculate IoU
            max_iou_per_gbox = [max([compute_iou(gbox, gt_box) for gt_box in gt_boxes], default=0) 
                                for gbox in gradcam_boxes]
            avg_iou = np.mean(max_iou_per_gbox) if max_iou_per_gbox else 0
            iou_scores.append(avg_iou)
    
    return np.mean(iou_scores) if iou_scores else 0

def fine_tune_model():
    """Fine-tune the model on the RSNA dataset."""
    # Load dataset
    image_dir = os.path.join(RSNA_DATA_DIR, "train_images")
    labels_file = os.path.join(RSNA_DATA_DIR, "labels", "stage_2_train_labels.csv")
    dataset = RSNADataset(image_dir, labels_file, transform=transform)
    
    # Split dataset into train/val/test
    indices = list(range(len(dataset)))
    train_indices, temp_indices = train_test_split(indices, test_size=0.2, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Debug: Print label distribution in each split
    train_labels = [dataset[i][1] for i in train_indices]
    val_labels = [dataset[i][1] for i in val_indices]
    test_labels = [dataset[i][1] for i in test_indices]
    print(f"Train label distribution: {np.bincount(train_labels)} (0=Normal, 1=Pneumonia)")
    print(f"Validation label distribution: {np.bincount(val_labels)} (0=Normal, 1=Pneumonia)")
    print(f"Test label distribution: {np.bincount(test_labels)} (0=Normal, 1=Pneumonia)")
    
    # Load the model
    model = PneumoniaClassifier(
        model_path=MODEL_PATH,
        target_layer="layer4[-1]",
        input_size=224,
        threshold=0.4,
        max_area_fraction=0.7,
        min_area_fraction=0.01,
        use_morph_ops=False,
        use_gradcam_plus_plus=True
    ).to(device)
    
    # Evaluate the current model
    print("Evaluating current model...")
    accuracy, precision, recall, f1 = evaluate_classification(model, val_loader)
    print(f"Current Model - Validation Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    labels_df = pd.read_csv(labels_file)
    val_iou = evaluate_localization(model, val_loader, labels_df)
    print(f"Current Model - Validation IoU: {val_iou:.4f}")
    
    # Fine-tuning setup
    # Freeze all layers except fc and layer4
    for name, param in model.named_parameters():
        if "fc" not in name and "layer4" not in name:
            param.requires_grad = False
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    # Fine-tuning loop
    num_epochs = 10
    best_val_accuracy = accuracy
    best_val_iou = val_iou
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels, _ in train_loader:  # Unpack 3 values, ignore patient_id
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Evaluate on validation set
        val_accuracy, val_precision, val_recall, val_f1 = evaluate_classification(model, val_loader)
        val_iou = evaluate_localization(model, val_loader, labels_df)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Training Loss: {running_loss/len(train_loader):.4f}")
        print(f"  Validation Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        print(f"  Validation IoU: {val_iou:.4f}")
        
        # Early stopping if accuracy drops too much
        if val_accuracy < best_val_accuracy * 0.95:  # Allow 5% drop
            print("Validation accuracy dropped too much. Stopping training.")
            break
        
        # Save best model based on IoU
        if val_iou > best_val_iou and val_accuracy >= best_val_accuracy * 0.95:
            best_val_iou = val_iou
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
            torch.save(best_model_state, FINETUNED_MODEL_PATH)
            print(f"Saved best model with IoU: {best_val_iou:.4f}")
    
    # Evaluate on test set
    print("Evaluating fine-tuned model on test set...")
    model.load_state_dict(torch.load(FINETUNED_MODEL_PATH))
    test_accuracy, test_precision, test_recall, test_f1 = evaluate_classification(model, test_loader)
    test_iou = evaluate_localization(model, test_loader, labels_df)
    print(f"Test Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
    print(f"Test IoU: {test_iou:.4f}")

if __name__ == "__main__":
    fine_tune_model()