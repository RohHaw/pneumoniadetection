"""
Training and evaluation script for the Pneumonia X-Ray Classifier on the RSNA dataset.

This module defines a PyTorch Dataset for loading RSNA Pneumonia Detection Challenge DICOM images,
a ResNet-50-based classifier, and functions for fine-tuning and evaluating the model. It includes
data preprocessing, model training with class-weighted loss, and comprehensive evaluation with
metrics and visualisations (confusion matrix, ROC curve, and probability histogram). Results are
saved to a text file and visualisations to PNG files.

Author: Rohman Hawrylak
Date: April 2025
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
import pydicom

class RSNADataset(Dataset):
    """
    PyTorch Dataset for the RSNA Pneumonia Detection Challenge.

    Loads DICOM chest X-ray images from specified directories and their corresponding labels
    from a CSV file. Normalises images, converts them to RGB, and applies optional transformations.
    Designed for use with the SimplePneumoniaClassifier for binary classification (Normal vs. Pneumonia).

    Attributes:
        image_dirs (list): List of directories containing DICOM image files.
        transform (callable, optional): Transformations to apply to images.
        df (pandas.DataFrame): DataFrame with patient IDs and labels.
        image_paths (list): List of paths to valid DICOM images.
        labels (list): List of corresponding labels (0=Normal, 1=Pneumonia).
    """
    def __init__(self, image_dirs, csv_path, transform=None):
        """
        Initialise the RSNADataset with image directories, labels file, and optional transform.

        Args:
            image_dirs (list): List of paths to directories containing DICOM image files.
            csv_path (str): Path to the CSV file containing patient IDs and labels.
            transform (callable, optional): Optional transform to apply to images.
        """
        self.image_dirs = image_dirs
        self.transform = transform
        self.df = pd.read_csv(csv_path)
        self.image_paths = []
        self.labels = []

        # Scan directories for valid DICOM images and labels
        print("Scanning image directories...")
        for image_dir in image_dirs:
            for img_file in os.listdir(image_dir):
                if img_file.endswith('.dcm'):
                    patient_id = img_file.split('.')[0]
                    row = self.df[self.df['patientId'] == patient_id]
                    if not row.empty:
                        label = row['Target'].iloc[0]
                        self.image_paths.append(os.path.join(image_dir, img_file))
                        self.labels.append(label)
            print(f"Found {len(self.image_paths)} images in {image_dir} so far...")

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of valid DICOM images with labels.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieve an image and its label by index.

        Loads a DICOM image, normalises it, converts it to RGB, applies transformations if specified,
        and returns the image tensor and its label. Returns a zero tensor if image loading fails.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label)
                - image (torch.Tensor): Preprocessed image tensor (or zero tensor on error).
                - label (int): Binary label (0=Normal, 1=Pneumonia).
        """
        img_path = self.image_paths[idx]
        try:
            # Load and preprocess DICOM image
            dcm = pydicom.dcmread(img_path)
            img_array = dcm.pixel_array
            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=2)  # Convert to RGB
            # Normalise to [0, 255]
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8) * 255
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_array)

            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            # Handle loading errors gracefully
            print(f"Error loading image {img_path}: {str(e)}")
            return torch.zeros((3, 224, 224)), self.labels[idx]

class SimplePneumoniaClassifier(nn.Module):
    """
    ResNet-50-based classifier for pneumonia detection.

    Uses a pretrained ResNet-50 model with a modified fully connected layer for binary
    classification (Normal vs. Pneumonia). Supports loading pretrained weights and includes
    a prediction method for single images with probability outputs.

    Attributes:
        device (torch.device): Device for computation (GPU or CPU).
        model (nn.Module): ResNet-50 model with custom FC layer.
        transform (callable): Image preprocessing transformations.
        classes (list): List of class names ['Normal', 'Pneumonia'].
        input_size (int): Input image size (height and width).
    """
    def __init__(self, model_path="Training/UCSD/model_UCSD.pth", input_size=224):
        """
        Initialise the SimplePneumoniaClassifier.

        Args:
            model_path (str, optional): Path to pretrained model weights. Defaults to
                "Training/UCSD/model_UCSD.pth".
            input_size (int, optional): Input image size (height and width). Defaults to 224.
        """
        super(SimplePneumoniaClassifier, self).__init__()
        
        # Set device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load ResNet-50 with custom FC layer
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.model.fc.in_features, 2)
        )

        # Load pretrained weights if available
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        
        self.model.to(self.device)

        # Define preprocessing transformations
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.classes = ['Normal', 'Pneumonia']
        self.input_size = input_size

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Model output logits.
        """
        return self.model(x)

    def predict(self, image):
        """
        Predict the class and probabilities for a single image.

        Args:
            image (PIL.Image): Input image for classification.

        Returns:
            dict: Prediction results with class and probabilities.
                - class (str): Predicted class ('Normal' or 'Pneumonia').
                - probabilities (dict): Probabilities for 'Normal' and 'Pneumonia' in percentage.
        """
        self.model.eval()
        # Preprocess and predict
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_class = np.argmax(probs)
        return {
            'class': self.classes[pred_class],
            'probabilities': {
                'Normal': probs[0] * 100,
                'Pneumonia': probs[1] * 100
            }
        }

def fine_tune_and_evaluate(classifier, dataset):
    """
    Fine-tune the classifier on the RSNA dataset and evaluate on a test set.

    Splits the dataset into training (60%), validation (20%), and test (20%) sets with
    stratified sampling. Fine-tunes the classifier's fully connected layer using a class-weighted
    loss function, evaluates performance on the validation set, and saves the best model.
    Finally, evaluates the model on the test set and generates metrics and visualisations.

    Args:
        classifier (SimplePneumoniaClassifier): The classifier to fine-tune.
        dataset (RSNADataset): The RSNA dataset for training and evaluation.

    Returns:
        dict: Evaluation metrics including accuracy, precision, recall, F1 score, ROC-AUC,
            confusion matrix, labels, and probabilities.
    """
    # Split dataset into train (60%), validation (20%), and test (20%)
    indices = list(range(len(dataset)))
    train_val_indices, test_indices = train_test_split(indices, test_size=0.2, stratify=dataset.labels, random_state=42)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=0.25, stratify=[dataset.labels[i] for i in train_val_indices], random_state=42)  # 0.25 of 80% = 20%
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Freeze all layers except the fully connected layer
    for name, param in classifier.model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

    # Set up optimiser and class-weighted loss
    optimizer = torch.optim.Adam(classifier.model.fc.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.44]).to(classifier.device))  # Weight for imbalance (77% Normal, 23% Pneumonia)
    num_epochs = 5

    print("Fine-tuning on RSNA train set...")
    best_val_acc = 0
    for epoch in range(num_epochs):
        classifier.model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(classifier.device), labels.to(classifier.device)
            optimizer.zero_grad()
            outputs = classifier.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate model
        classifier.model.eval()
        val_labels = []
        val_preds = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(classifier.device), labels.to(classifier.device)
                outputs = classifier.model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds)
        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss/len(train_loader):.4f}, Val Accuracy: {val_acc:.4f}")

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.model.state_dict(), "Training/UCSD/model_UCSD_finetuned_rsna.pth")
            print("Saved best model based on validation accuracy")

    # Load best model for evaluation
    classifier.model.load_state_dict(torch.load("Training/UCSD/model_UCSD_finetuned_rsna.pth", map_location=classifier.device, weights_only=True))

    # Evaluate on test set
    all_labels = []
    all_preds = []
    all_probs = []
    sample_count = 0

    print("Evaluating on RSNA test set...")
    for images, labels in test_loader:
        for img, label in zip(images, labels):
            # Denormalise image for prediction
            img_np = img.numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = (img_np * std + mean) * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            # Predict and collect results
            result = classifier.predict(img_pil)
            pred_class = 1 if result['probabilities']['Pneumonia'] > 50 else 0
            all_labels.append(label.item())
            all_preds.append(pred_class)
            all_probs.append(result['probabilities']['Pneumonia'] / 100)

            # Log first few samples for debugging
            sample_count += 1
            if sample_count <= 5:
                print(f"Sample {sample_count}: True={classifier.classes[label.item()]}, "
                      f"Pred={classifier.classes[pred_class]}, "
                      f"Probs={{'Normal': {result['probabilities']['Normal']:.2f}, "
                      f"'Pneumonia': {result['probabilities']['Pneumonia']:.2f}}}")

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    roc_auc = roc_auc_score(all_labels, all_probs)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Generate probability histogram
    plt.figure(figsize=(10, 6))
    plt.hist([p for p, l in zip(all_probs, all_labels) if l == 0], bins=50, alpha=0.5, label='Normal', color='blue')
    plt.hist([p for p, l in zip(all_probs, all_labels) if l == 1], bins=50, alpha=0.5, label='Pneumonia', color='red')
    plt.xlabel('Pneumonia Probability')
    plt.ylabel('Frequency')
    plt.title('Probability Distribution - RSNA Test Set (Fine-Tuned)')
    plt.legend()
    plt.savefig('probability_histogram_rsna_finetuned.png')
    plt.close()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix,
        'all_labels': all_labels,
        'all_probs': all_probs
    }

def plot_confusion_matrix(conf_matrix, class_names=['Normal', 'Pneumonia']):
    """
    Plot and save a confusion matrix for the RSNA dataset.

    Args:
        conf_matrix (numpy.ndarray): Confusion matrix from evaluation.
        class_names (list, optional): List of class names. Defaults to ['Normal', 'Pneumonia'].
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - RSNA Dataset (Fine-Tuned)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_rsna.png')
    plt.close()

def plot_roc_curve(labels, probs):
    """
    Plot and save an ROC curve for the RSNA dataset.

    Args:
        labels (numpy.ndarray): True labels from the test set.
        probs (numpy.ndarray): Predicted probabilities for the positive class.
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(labels, probs):.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - RSNA Dataset (Fine-Tuned)')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_rsna.png')
    plt.close()

def main():
    """
    Main function to execute fine-tuning and evaluation on the RSNA dataset.

    Loads the RSNA dataset, initialises the classifier, fine-tunes the model, evaluates
    performance, and saves results and visualisations to files.
    """
    # Define paths and parameters
    train_dir = "D:/rsna_data/train_images"
    test_dir = "D:/rsna_data/test_images"
    csv_path = "D:/rsna_data/labels/stage_2_train_labels.csv"
    model_path = "Training/UCSD/model_UCSD.pth"
    output_file = "evaluation_results_rsna.txt"

    # Define preprocessing transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    print("Loading RSNA dataset...")
    dataset = RSNADataset([train_dir, test_dir], csv_path, transform=transform)

    # Log dataset statistics
    print(f"Dataset size: {len(dataset)} images")
    print(f"Class distribution: {{'Normal': {sum(1 for l in dataset.labels if l == 0)}, 'Pneumonia': {sum(1 for l in dataset.labels if l == 1)}}}")

    # Initialise classifier
    classifier = SimplePneumoniaClassifier(model_path=model_path)

    # Fine-tune and evaluate
    print("Fine-tuning and evaluating model on RSNA dataset...")
    metrics = fine_tune_and_evaluate(classifier, dataset)

    # Save evaluation results
    with open(output_file, 'w') as f:
        f.write("Evaluation Results on RSNA Dataset (Fine-Tuned)\n")
        f.write("==============================================\n\n")
        f.write(f"Train Images Path: {train_dir}\n")
        f.write(f"Test Images Path: {test_dir}\n")
        f.write(f"Labels CSV: {csv_path}\n")
        f.write(f"Total Images: {len(dataset)}\n")
        f.write(f"Class Distribution: {{'Normal': {sum(1 for l in dataset.labels if l == 0)}, 'Pneumonia': {sum(1 for l in dataset.labels if l == 1)}}}\n\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"ROC-AUC: {metrics['roc_auc']:.4f}\n")
        f.write(f"Confusion Matrix:\n{metrics['confusion_matrix'].tolist()}\n")

    print(f"Results saved to {output_file}")
    # Generate visualisations
    print("Generating visualizations...")
    plot_confusion_matrix(metrics['confusion_matrix'])
    plot_roc_curve(metrics['all_labels'], metrics['all_probs'])
    print("Visualizations saved as confusion_matrix_rsna.png, roc_curve_rsna.png, and probability_histogram_rsna_finetuned.png")

if __name__ == "__main__":
    main()