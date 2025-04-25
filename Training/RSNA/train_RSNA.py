"""
Training script for fine-tuning a ResNet-50 model on the RSNA Pneumonia Detection Challenge dataset.

This module implements a comprehensive training pipeline for a ResNet-50-based pneumonia classifier.
It includes data loading with augmentation, a custom dataset for DICOM images, focal loss for handling
class imbalance, mixed precision training, early stopping, and cosine annealing learning rate scheduling.
The script trains the model in two stages (frozen backbone and full fine-tuning), evaluates performance
on validation and test sets, and generates visualisations for training history, confusion matrix, and
ROC curve. Results are saved to a text file and models to checkpoint files.

Author: Rohman Hawrylak
Date: April 2025
"""

import os
import pandas as pd
import numpy as np
import pydicom
from torchvision import models
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Config:
    """
    Configuration class for training parameters.

    Defines paths, hyperparameters, and settings for the training pipeline, including data directories,
    model architecture, training schedule, and device settings.

    Attributes:
        TRAIN_DATA_DIR (str): Directory for training DICOM images.
        TEST_DATA_DIR (str): Directory for test DICOM images.
        CSV_PATH (str): Path to the CSV file with training labels.
        IMAGE_SIZE (int): Size for resizing images (height and width).
        BATCH_SIZE (int): Batch size for training and evaluation.
        NUM_CLASSES (int): Number of output classes (Normal and Pneumonia).
        EPOCHS (int): Total number of training epochs.
        BASE_LR (float): Base learning rate for the optimiser.
        WEIGHT_DECAY (float): Weight decay for regularisation.
        DROPOUT (float): Dropout probability in the fully connected layer.
        USE_AMP (bool): Enable mixed precision training.
        PATIENCE (int): Patience for early stopping.
        MIN_DELTA (float): Minimum improvement for early stopping.
        DEVICE (torch.device): Device for computation (GPU or CPU).
        TRAIN_SPLIT (float): Proportion of data for training.
        VAL_SPLIT (float): Proportion of data for validation.
    """
    TRAIN_DATA_DIR = "/vol/scratch/SoC/misc/2024/sc21r2h/rsna_data/train_images"
    TEST_DATA_DIR = "/vol/scratch/SoC/misc/2024/sc21r2h/rsna_data/test_images"
    CSV_PATH = "/vol/scratch/SoC/misc/2024/sc21r2h/rsna_data/labels/stage_2_train_labels.csv"
    IMAGE_SIZE = 224
    BATCH_SIZE = 64
    NUM_CLASSES = 2
    EPOCHS = 30
    BASE_LR = 1e-4
    WEIGHT_DECAY = 0.0001
    DROPOUT = 0.3
    USE_AMP = True
    PATIENCE = 10
    MIN_DELTA = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.2

class TrainingHistory:
    """
    Track training and validation metrics during model training.

    Stores losses, accuracies, F1 scores, and ROC-AUC scores for each epoch to monitor performance
    and generate visualisations.

    Attributes:
        train_losses (list): Training loss per epoch.
        val_losses (list): Validation loss per epoch.
        train_accuracies (list): Training accuracy per epoch.
        val_accuracies (list): Validation accuracy per epoch.
        val_f1s (list): Validation F1 score per epoch.
        val_roc_aucs (list): Validation ROC-AUC score per epoch.
    """
    def __init__(self):
        """Initialise empty lists for tracking metrics."""
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_f1s = []
        self.val_roc_aucs = []

def plot_training_history(history):
    """
    Plot and save training and validation metrics.

    Generates a 2x2 subplot with training/validation loss, accuracy, validation F1 score,
    and validation ROC-AUC score over epochs.

    Args:
        history (TrainingHistory): Object containing training and validation metrics.
    """
    plt.figure(figsize=(15, 10))
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history.train_losses, label='Train Loss')
    plt.plot(history.val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history.train_accuracies, label='Train Accuracy')
    plt.plot(history.val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot F1 score
    plt.subplot(2, 2, 3)
    plt.plot(history.val_f1s, label='Validation F1')
    plt.title('Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    # Plot ROC-AUC
    plt.subplot(2, 2, 4)
    plt.plot(history.val_roc_aucs, label='Validation ROC-AUC')
    plt.title('Validation ROC-AUC Score')
    plt.xlabel('Epoch')
    plt.ylabel('ROC-AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('rsna_training_history.png')
    plt.close()

def plot_confusion_matrix(conf_matrix, class_names=['Normal', 'Pneumonia']):
    """
    Plot and save a confusion matrix.

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
    plt.savefig('rsna_confusion_matrix.png')
    plt.close()

def plot_roc_curve(all_labels, all_probs):
    """
    Plot and save an ROC curve.

    Args:
        all_labels (numpy.ndarray): True labels from the validation set.
        all_probs (numpy.ndarray): Predicted probabilities for the positive class.
    """
    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('rsna_roc_curve.png')
    plt.close()

# Data Transforms
train_transform = A.Compose([
    A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),  # Resize images
    A.HorizontalFlip(p=0.5),  # Random horizontal flip
    A.VerticalFlip(p=0.5),  # Random vertical flip
    A.Rotate(limit=15, p=0.5),  # Random rotation
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),  # Random shift, scale, rotate
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Adjust brightness/contrast
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),  # Add Gaussian noise
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalise with ImageNet stats
    ToTensorV2()  # Convert to tensor
])

eval_transform = A.Compose([
    A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),  # Resize images
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalise with ImageNet stats
    ToTensorV2()  # Convert to tensor
])

class EarlyStopping:
    """
    Implement early stopping based on validation loss.

    Stops training if the validation loss does not improve by a minimum delta after a specified
    number of epochs.

    Attributes:
        patience (int): Number of epochs to wait for improvement.
        min_delta (float): Minimum improvement in validation loss.
        counter (int): Number of epochs since last improvement.
        best_loss (float): Best validation loss observed.
        early_stop (bool): Flag indicating whether to stop training.
    """
    def __init__(self, patience=Config.PATIENCE, min_delta=Config.MIN_DELTA):
        """
        Initialise EarlyStopping with patience and minimum delta.

        Args:
            patience (int, optional): Number of epochs to wait. Defaults to Config.PATIENCE.
            min_delta (float, optional): Minimum improvement. Defaults to Config.MIN_DELTA.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Check if early stopping should be triggered.

        Args:
            val_loss (float): Current validation loss.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class RSNADataset(Dataset):
    """
    PyTorch Dataset for the RSNA Pneumonia Detection Challenge training data.

    Loads DICOM chest X-ray images and their labels from a DataFrame, normalises them, converts
    to RGB, and applies transformations. Designed for training and validation.

    Attributes:
        df (pandas.DataFrame): DataFrame with patient IDs and labels.
        data_dir (str): Directory containing DICOM images.
        transform (callable, optional): Transformations to apply to images.
    """
    def __init__(self, df, data_dir, transform=None):
        """
        Initialise the RSNADataset with DataFrame, data directory, and transform.

        Args:
            df (pandas.DataFrame): DataFrame with patient IDs and labels.
            data_dir (str): Path to the directory containing DICOM images.
            transform (callable, optional): Optional transform to apply to images.
        """
        self.df = df
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of samples in the DataFrame.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieve an image and its label by index.

        Loads a DICOM image, normalises it, converts to RGB, applies transformations, and
        returns the image tensor and label.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label)
                - image (torch.Tensor): Preprocessed image tensor.
                - label (int): Binary label (0=Normal, 1=Pneumonia).
        """
        # Load image and label
        pid = self.df.iloc[idx]['patientId']
        label = self.df.iloc[idx]['Target']
        path = os.path.join(self.data_dir, f"{pid}.dcm")
        dicom = pydicom.dcmread(path)
        image = dicom.pixel_array
        # Normalise to [0, 255]
        image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
        image = image.astype(np.uint8)
        # Convert grayscale to RGB
        image = np.stack([image] * 3, axis=-1)  # Shape: (224, 224, 3)
        if self.transform:
            image = self.transform(image=image)['image']
        return image, label

class RSNATestDataset(Dataset):
    """
    PyTorch Dataset for the RSNA Pneumonia Detection Challenge test data.

    Loads DICOM chest X-ray images from a directory, normalises them, converts to RGB, and
    applies transformations. Returns placeholder labels for compatibility with evaluation.

    Attributes:
        data_dir (str): Directory containing DICOM images.
        transform (callable, optional): Transformations to apply to images.
        image_files (list): List of DICOM image filenames.
    """
    def __init__(self, data_dir, transform=None):
        """
        Initialise the RSNATestDataset with data directory and transform.

        Args:
            data_dir (str): Path to the directory containing DICOM images.
            transform (callable, optional): Optional transform to apply to images.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.dcm')]

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of DICOM images in the directory.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Retrieve an image and a placeholder label by index.

        Loads a DICOM image, normalises it, converts to RGB, applies transformations, and
        returns the image tensor with a placeholder label (-1).

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label)
                - image (torch.Tensor): Preprocessed image tensor.
                - label (int): Placeholder label (-1).
        """
        # Load image
        path = os.path.join(self.data_dir, self.image_files[idx])
        dicom = pydicom.dcmread(path)
        image = dicom.pixel_array
        # Normalise to [0, 255]
        image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
        image = image.astype(np.uint8)
        # Convert grayscale to RGB
        image = np.stack([image] * 3, axis=-1)  # Shape: (224, 224, 3)
        if self.transform:
            image = self.transform(image=image)['image']
        return image, -1  # Placeholder label

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Modifies cross-entropy loss to focus on hard-to-classify samples by down-weighting
    easy examples.

    Attributes:
        alpha (float): Weighting factor for the loss.
        gamma (float): Focusing parameter for hard examples.
        reduction (str): Reduction method ('mean', 'sum', or 'none').
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Initialise FocalLoss with alpha, gamma, and reduction method.

        Args:
            alpha (float, optional): Weighting factor. Defaults to 1.
            gamma (float, optional): Focusing parameter. Defaults to 2.
            reduction (str, optional): Reduction method. Defaults to 'mean'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute the focal loss.

        Args:
            inputs (torch.Tensor): Model output logits.
            targets (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Computed focal loss.
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def train(model, dataloader, optimizer, criterion, device, scaler):
    """
    Train the model for one epoch.

    Uses mixed precision training and computes loss and accuracy for the training set.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (optim.Optimizer): Optimiser for updating model weights.
        criterion (nn.Module): Loss function.
        device (torch.device): Device for computation.
        scaler (GradScaler): Scaler for mixed precision training.

    Returns:
        tuple: (train_loss, train_accuracy)
            - train_loss (float): Average training loss.
            - train_accuracy (float): Training accuracy.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Use mixed precision training
        with torch.amp.autocast(device_type='cuda', enabled=Config.USE_AMP):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # Backpropagate with scaled loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        # Log progress every 10 batches
        if i % 10 == 0:
            print(f"\rBatch {i}/{len(dataloader)}: Loss = {loss.item():.4f}", end="")
    
    accuracy = correct / total
    return running_loss / len(dataloader), accuracy

def evaluate(model, dataloader, criterion, device, phase="val"):
    """
    Evaluate the model on a validation or test set.

    Computes loss (for validation), predictions, probabilities, and metrics including accuracy,
    precision, recall, F1 score, ROC-AUC, and confusion matrix.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for evaluation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device for computation.
        phase (str, optional): Evaluation phase ('val' or 'test'). Defaults to 'val'.

    Returns:
        dict: Evaluation metrics or predictions/probabilities for test phase.
    """
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.amp.autocast(device_type='cuda', enabled=Config.USE_AMP):
                outputs = model(inputs)
                if phase != "test":
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    if phase != "test":
        # Compute validation metrics
        accuracy = (all_predictions == all_labels).mean()
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        return {
            'loss': running_loss / len(dataloader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': conf_matrix,
            'all_labels': all_labels,
            'all_probs': all_probs
        }
    else:
        # Return predictions and probabilities for test set
        return {'all_predictions': all_predictions, 'all_probs': all_probs}

def main():
    """
    Main function to execute the training and evaluation pipeline.

    Loads the RSNA dataset, splits it into training and validation sets, initialises the ResNet-50
    model, trains in two stages (frozen backbone and full fine-tuning), evaluates on the test set,
    and saves results and visualisations.
    """
    # Load and split dataset
    print("Loading training dataset...")
    df = pd.read_csv(Config.CSV_PATH)
    total_size = len(df)
    train_size = int(Config.TRAIN_SPLIT * total_size)
    val_size = total_size - train_size
    
    # Split DataFrame indices
    train_subset, val_subset = random_split(
        df, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_df = df.iloc[train_subset.indices].reset_index(drop=True)
    val_df = df.iloc[val_subset.indices].reset_index(drop=True)
    
    # Initialise datasets and loaders
    train_dataset = RSNADataset(train_df, Config.TRAIN_DATA_DIR, train_transform)
    val_dataset = RSNADataset(val_df, Config.TRAIN_DATA_DIR, eval_transform)
    test_dataset = RSNATestDataset(Config.TEST_DATA_DIR, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Initialise model
    print("Setting up model...")
    model = models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Dropout(p=Config.DROPOUT),
        nn.Linear(model.fc.in_features, Config.NUM_CLASSES)
    )
    model = model.to(Config.DEVICE)

    # Set up training components
    criterion = FocalLoss(alpha=1, gamma=2)
    scaler = GradScaler(enabled=Config.USE_AMP)
    history = TrainingHistory()
    early_stopping = EarlyStopping()
    best_val_f1 = 0.0
    best_epoch = 0

    print(f"\nTraining with {len(train_dataset)} samples, validating with {len(val_dataset)} samples, testing with {len(test_dataset)} samples")

    # Stage 1: Train only the classifier (frozen backbone)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.BASE_LR, weight_decay=Config.WEIGHT_DECAY)
    for epoch in range(5):
        print(f"\nEpoch {epoch+1}/5 (Stage 1)")
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, Config.DEVICE, scaler)
        print(f"\nTrain Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        val_metrics = evaluate(model, val_loader, criterion, Config.DEVICE, "val")
        print(f"Validation Loss: {val_metrics['loss']:.4f}")
        print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Validation F1: {val_metrics['f1']:.4f}")
        print(f"Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")
        history.train_losses.append(train_loss)
        history.train_accuracies.append(train_acc)
        history.val_losses.append(val_metrics['loss'])
        history.val_accuracies.append(val_metrics['accuracy'])
        history.val_f1s.append(val_metrics['f1'])
        history.val_roc_aucs.append(val_metrics['roc_auc'])
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            torch.save(model.state_dict(), "best_rsna_model_v2_stage1.pth")
            print("New best model saved (Stage 1)!")

    # Stage 2: Fine-tune all layers
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=Config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS - 5)
    for epoch in range(Config.EPOCHS - 5):
        print(f"\nEpoch {epoch+6}/{Config.EPOCHS} (Stage 2)")
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, Config.DEVICE, scaler)
        print(f"\nTrain Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        val_metrics = evaluate(model, val_loader, criterion, Config.DEVICE, "val")
        print(f"Validation Loss: {val_metrics['loss']:.4f}")
        print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Validation F1: {val_metrics['f1']:.4f}")
        print(f"Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")
        scheduler.step()
        history.train_losses.append(train_loss)
        history.train_accuracies.append(train_acc)
        history.val_losses.append(val_metrics['loss'])
        history.val_accuracies.append(val_metrics['accuracy'])
        history.val_f1s.append(val_metrics['f1'])
        history.val_roc_aucs.append(val_metrics['roc_auc'])
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch + 5
            torch.save(model.state_dict(), "best_rsna_model_v2.pth")
            print("New best model saved (Stage 2)!")
        early_stopping(val_metrics['loss'])
        if early_stopping.early_stop:
            print(f"Early stopping triggered after epoch {epoch+6}")
            break

    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load("best_rsna_model_v2.pth"))
    test_metrics = evaluate(model, test_loader, criterion, Config.DEVICE, "test")

    # Save visualisations
    print("\nGenerating visualization plots...")
    plot_training_history(history)
    plot_confusion_matrix(val_metrics['confusion_matrix'])
    plot_roc_curve(val_metrics['all_labels'], val_metrics['all_probs'])

    # Save results
    with open('rsna_model_v2_results.txt', 'w') as f:
        f.write("RSNA Model Results\n")
        f.write("==================\n\n")
        f.write(f"Train samples: {len(train_dataset)}\n")
        f.write(f"Validation samples: {len(val_dataset)}\n")
        f.write(f"Test samples: {len(test_dataset)}\n\n")
        f.write(f"Best Validation F1: {best_val_f1:.4f} (Epoch {best_epoch+1})\n\n")
        f.write("Validation Metrics:\n")
        f.write(f"Accuracy: {val_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {val_metrics['precision']:.4f}\n")
        f.write(f"Recall: {val_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {val_metrics['f1']:.4f}\n")
        f.write(f"ROC-AUC: {val_metrics['roc_auc']:.4f}\n")

    print("\nTraining completed! Results saved to rsna_model_v2_results.txt")
    print("Model saved to best_rsna_v2_model.pth")

if __name__ == "__main__":
    main()