"""
Evaluation script for the Pneumonia X-Ray Classifier on a new dataset.

This module defines a ResNet-50-based classifier for pneumonia detection and evaluates its performance
on a new dataset using a fine-tuned model. It computes metrics including accuracy, precision, recall,
F1 score, ROC-AUC, and confusion matrix, and generates visualisations for the confusion matrix and
ROC curve. Results are saved to a text file and visualisations to PNG files.

Author: Rohman Hawrylak
Date: April 2025
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

class SimplePneumoniaClassifier(nn.Module):
    """
    ResNet-50-based classifier for pneumonia detection.

    Uses a ResNet-50 model with a modified fully connected layer for binary classification
    (Normal vs. Pneumonia). Loads fine-tuned weights and provides a method for predicting
    class probabilities on single images.

    Attributes:
        device (torch.device): Device for computation (GPU or CPU).
        model (nn.Module): ResNet-50 model with custom FC layer.
        transform (callable): Image preprocessing transformations.
        classes (list): List of class names ['Normal', 'Pneumonia'].
        input_size (int): Input image size (height and width).
    """
    def __init__(self, model_path="Training/UCSD/model_UCSD_finetuned_rsna.pth", input_size=224):
        """
        Initialise the SimplePneumoniaClassifier.

        Args:
            model_path (str, optional): Path to fine-tuned model weights. Defaults to
                "Training/UCSD/model_UCSD_finetuned_rsna.pth".
            input_size (int, optional): Input image size (height and width). Defaults to 224.
        """
        super(SimplePneumoniaClassifier, self).__init__()
        
        # Set device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load ResNet-50 with custom FC layer
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3),  # Dropout for regularisation
            nn.Linear(self.model.fc.in_features, 2)  # Binary classification
        )

        # Load fine-tuned weights if available
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

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

def evaluate_model(classifier, dataloader):
    """
    Evaluate the classifier on a dataset.

    Computes metrics including accuracy, precision, recall, F1 score, ROC-AUC, and confusion
    matrix by processing images individually and comparing predictions to true labels.

    Args:
        classifier (SimplePneumoniaClassifier): The classifier to evaluate.
        dataloader (DataLoader): DataLoader for the evaluation dataset.

    Returns:
        dict: Evaluation metrics including accuracy, precision, recall, F1 score, ROC-AUC,
            confusion matrix, labels, and probabilities.
    """
    all_labels = []
    all_preds = []
    all_probs = []

    # Define mean and std for denormalisation
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for images, labels in dataloader:
        for img, label in zip(images, labels):
            # Denormalise image for prediction
            img_np = img.numpy().transpose(1, 2, 0)
            img_np = (img_np * std + mean) * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            # Predict and collect results
            result = classifier.predict(img_pil)
            pred_class = classifier.classes.index(result['class'])
            all_labels.append(label.item())
            all_preds.append(pred_class)
            all_probs.append(result['probabilities']['Pneumonia'] / 100)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    roc_auc = roc_auc_score(all_labels, all_probs)
    conf_matrix = confusion_matrix(all_labels, all_preds)

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
    Plot and save a confusion matrix for the evaluation results.

    Args:
        conf_matrix (numpy.ndarray): Confusion matrix from evaluation.
        class_names (list, optional): List of class names. Defaults to ['Normal', 'Pneumonia'].
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_new_dataset_finetuned.png')
    plt.close()

def plot_roc_curve(labels, probs):
    """
    Plot and save an ROC curve for the evaluation results.

    Args:
        labels (numpy.ndarray): True labels from the dataset.
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
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_new_dataset_finetuned.png')
    plt.close()

def main():
    """
    Main function to execute evaluation on a new dataset.

    Loads the dataset, initialises the classifier with fine-tuned weights, evaluates performance,
    and saves results and visualisations to files.
    """
    # Define paths and parameters
    data_dir = "D:/archive_combined"
    model_path = "Training/UCSD/model_UCSD_finetuned_rsna.pth"
    output_file = "evaluation_results_new_dataset_finetuned.txt"

    # Define preprocessing transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    print("Loading dataset...")
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Log dataset statistics
    print(f"Dataset size: {len(dataset)} images")
    print(f"Class distribution: {dict(zip(dataset.classes, [len([x for x in dataset.targets if x == i]) for i in range(len(dataset.classes))]))}")

    # Initialise classifier
    classifier = SimplePneumoniaClassifier(model_path=model_path)

    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(classifier, dataloader)

    # Save evaluation results
    with open(output_file, 'w') as f:
        f.write("Evaluation Results on New Dataset\n")
        f.write("================================\n\n")
        f.write(f"Dataset Path: {data_dir}\n")
        f.write(f"Total Images: {len(dataset)}\n")
        f.write(f"Class Distribution: {dict(zip(dataset.classes, [len([x for x in dataset.targets if x == i]) for i in range(len(dataset.classes))]))}\n\n")
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
    print("Visualizations saved as confusion_matrix_new_dataset.png, roc_curve_new_dataset.png")

if __name__ == "__main__":
    main()