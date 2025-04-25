"""
Module for training and validating a chest X-ray classifier.

This module defines a custom dataset and a ResNet-18-based validator for distinguishing chest
X-ray images (normal or pneumonia) from non-X-ray images. It includes functionality for dataset
loading, training with train/validation/test splits, evaluation with performance metrics, and
visualisation of confusion matrix and ROC curve. The trained model can also validate individual
images to determine if they are X-rays.

Author: Rohman Hawrylak
Date: April 2025
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

class ChestXrayValidationDataset(Dataset):
    """
    Custom Dataset for chest X-ray validation.

    Loads X-ray and non-X-ray images from specified directories, assigns labels (1 for X-ray,
    0 for non-X-ray), and applies transformations. Balances the dataset by sampling non-X-ray
    images to match the number of X-ray images.

    Attributes:
        transform (callable, optional): Transformations to apply to images.
        images (list): List of image file paths.
        labels (list): List of corresponding labels (1=X-ray, 0=non-X-ray).
    """
    def __init__(self, xray_dirs, non_xray_dir, transform=None):
        """
        Initialise the dataset with X-ray and non-X-ray directories and transform.

        Args:
            xray_dirs (list): List of directories containing X-ray images.
            non_xray_dir (str): Directory containing non-X-ray images.
            transform (callable, optional): Optional transform to apply to images.
        """
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load X-ray images
        for xray_dir in xray_dirs:
            for root, _, files in os.walk(xray_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(root, file))
                        self.labels.append(1)  # Label X-ray images as 1
                        
        # Load non-X-ray images and balance dataset
        non_xray_files = []
        for root, _, files in os.walk(non_xray_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    non_xray_files.append(os.path.join(root, file))
        
        # Randomly sample non-X-ray images to match X-ray count
        selected_non_xrays = random.sample(non_xray_files, min(len(self.labels), len(non_xray_files)))
        self.images.extend(selected_non_xrays)
        self.labels.extend([0] * len(selected_non_xrays))  # Label non-X-ray images as 0

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieve an image and its label by index.

        Loads an image, applies transformations, and returns the image tensor and label.
        Returns a zero tensor if image loading fails.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label)
                - image (torch.Tensor): Preprocessed image tensor (or zero tensor on error).
                - label (int): Binary label (1=X-ray, 0=non-X-ray).
        """
        image_path = self.images[idx]
        try:
            # Load and convert image to RGB
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            # Handle loading errors gracefully
            print(f"Error loading image {image_path}: {str(e)}")
            return torch.zeros((3, 224, 224)), self.labels[idx]

class ChestXrayValidator:
    """
    ResNet-18-based validator for distinguishing chest X-ray images from non-X-ray images.

    Uses a pretrained ResNet-18 model with a modified fully connected layer for binary
    classification. Supports training, evaluation with visualisation, and validation of
    individual images.

    Attributes:
        device (torch.device): Device for computation (GPU or CPU).
        model (nn.Module): ResNet-18 model with custom FC layer.
        transform (callable): Image preprocessing transformations.
    """
    def __init__(self, model_path=None):
        """
        Initialise the ChestXrayValidator.

        Args:
            model_path (str, optional): Path to pretrained model weights. Defaults to None.
        """
        # Set device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load ResNet-18 with custom FC layer
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),  # Dropout for regularisation
            nn.Linear(self.model.fc.in_features, 2)  # Binary classification
        )
        
        # Load pretrained weights if provided
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        
        # Define preprocessing transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def train(self, xray_dirs, non_xray_dir, epochs=10, batch_size=32, save_path='Training/Validator/xray_validator2.pth'):
        """
        Train the validator model on X-ray and non-X-ray images.

        Splits the dataset into training (70%), validation (15%), and test (15%) sets, trains
        the model, and saves the best model based on validation accuracy. Evaluates the model
        on the test set and generates visualisations.

        Args:
            xray_dirs (list): List of directories containing X-ray images.
            non_xray_dir (str): Directory containing non-X-ray images.
            epochs (int, optional): Number of training epochs. Defaults to 10.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            save_path (str, optional): Path to save the best model. Defaults to
                'Training/Validator/xray_validator2.pth'.
        """
        # Prepare dataset with train/val/test split
        dataset = ChestXrayValidationDataset(xray_dirs, non_xray_dir, self.transform)
        train_size = int(0.7 * len(dataset))  # 70% for training
        val_size = int(0.15 * len(dataset))  # 15% for validation
        test_size = len(dataset) - train_size - val_size  # 15% for testing
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Set up training components
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        best_val_acc = 0
        self.model.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Training loop with progress bar
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({'loss': running_loss/len(train_loader), 
                                'acc': 100 * train_correct/train_total})
            
            # Validation phase
            self.model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            print(f'Validation Accuracy: {val_acc:.2f}%')
            
            # Save best model based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f'Model saved with validation accuracy: {val_acc:.2f}%')
            
            self.model.train()
        
        # Evaluate on test set and generate visualisations
        self.evaluate_and_visualize(test_loader, save_path)

    def evaluate_and_visualize(self, test_loader, save_path):
        """
        Evaluate the model on the test set and generate visualisations.

        Computes accuracy, precision, recall, F1 score, confusion matrix, and ROC curve for the
        test set. Saves metrics to a text file and visualisations to PNG files.

        Args:
            test_loader (DataLoader): DataLoader for the test set.
            save_path (str): Path where the model was saved (used for context).
        """
        self.model.eval()
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                # Collect predictions and probabilities
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probabilities[:, 1].cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        # Save metrics to text file
        os.makedirs('evaluation_images_and_data', exist_ok=True)
        with open('evaluation_images_and_data/validator_accuracy/validator_metrics.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy:.4f}\n')
            f.write(f'Precision: {precision:.4f}\n')
            f.write(f'Recall: {recall:.4f}\n')
            f.write(f'F1-Score: {f1:.4f}\n')
        
        # Generate confusion matrix visualisation
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xticks([0, 1], ['Non-Xray', 'Xray'])
        plt.yticks([0, 1], ['Non-Xray', 'Xray'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha='center', va='center')
        plt.savefig('evaluation_images_and_data/validator_accuracy/validator_confusion_matrix.png')
        plt.close()
        
        # Generate ROC curve visualisation
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('evaluation_images_and_data/validator_accuracy/validator_roc_curve.png')
        plt.close()

    def validate_image(self, image):
        """
        Validate whether a single image is a chest X-ray.

        Args:
            image (PIL.Image): Input image to validate.

        Returns:
            tuple: (is_xray, confidence)
                - is_xray (bool): True if the image is classified as an X-ray.
                - confidence (float): Confidence score for X-ray classification (percentage).
        """
        self.model.eval()
        with torch.no_grad():
            # Preprocess and predict
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            is_xray = probabilities[0][1].item() > 0.5
            confidence = probabilities[0][1].item() * 100
            return is_xray, confidence