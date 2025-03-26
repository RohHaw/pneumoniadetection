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

# Configuration
class Config:
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

# Training History Tracker
class TrainingHistory:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_f1s = []
        self.val_roc_aucs = []

# Visualization Functions
def plot_training_history(history):
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(history.train_losses, label='Train Loss')
    plt.plot(history.val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(history.train_accuracies, label='Train Accuracy')
    plt.plot(history.val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(history.val_f1s, label='Validation F1')
    plt.title('Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
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
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('rsna_confusion_matrix.png')
    plt.close()

def plot_roc_curve(all_labels, all_probs):
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
    A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

eval_transform = A.Compose([
    A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=Config.PATIENCE, min_delta=Config.MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# RSNA Dataset Class
class RSNADataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pid = self.df.iloc[idx]['patientId']
        label = self.df.iloc[idx]['Target']
        path = os.path.join(self.data_dir, f"{pid}.dcm")
        dicom = pydicom.dcmread(path)
        image = dicom.pixel_array
        image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
        image = image.astype(np.uint8)
        # Convert grayscale to RGB by duplicating the channel
        image = np.stack([image] * 3, axis=-1)  # Shape: (224, 224, 3)
        if self.transform:
            image = self.transform(image=image)['image']
        return image, label

# Test Dataset Class
class RSNATestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.dcm')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.image_files[idx])
        dicom = pydicom.dcmread(path)
        image = dicom.pixel_array
        image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
        image = image.astype(np.uint8)
        # Convert grayscale to RGB by duplicating the channel
        image = np.stack([image] * 3, axis=-1)  # Shape: (224, 224, 3)
        if self.transform:
            image = self.transform(image=image)['image']
        return image, -1  # Placeholder label

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Training Function
def train(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type='cuda', enabled=Config.USE_AMP):  # Updated syntax
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        if i % 10 == 0:
            print(f"\rBatch {i}/{len(dataloader)}: Loss = {loss.item():.4f}", end="")
    
    accuracy = correct / total
    return running_loss / len(dataloader), accuracy

# Evaluation Function
def evaluate(model, dataloader, criterion, device, phase="val"):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.amp.autocast(device_type='cuda', enabled=Config.USE_AMP):  # Updated syntax
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
        return {'all_predictions': all_predictions, 'all_probs': all_probs}

# Main Function
def main():
    # Load and Split Dataset
    print("Loading training dataset...")
    df = pd.read_csv(Config.CSV_PATH)
    total_size = len(df)
    train_size = int(Config.TRAIN_SPLIT * total_size)
    val_size = total_size - train_size
    
    train_subset, val_subset = random_split(
        df, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_df = df.iloc[train_subset.indices].reset_index(drop=True)
    val_df = df.iloc[val_subset.indices].reset_index(drop=True)
    
    train_dataset = RSNADataset(train_df, Config.TRAIN_DATA_DIR, train_transform)
    val_dataset = RSNADataset(val_df, Config.TRAIN_DATA_DIR, eval_transform)
    test_dataset = RSNATestDataset(Config.TEST_DATA_DIR, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Model Setup
    print("Setting up model...")
    model = models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Dropout(p=Config.DROPOUT),
        nn.Linear(model.fc.in_features, Config.NUM_CLASSES)
    )
    model = model.to(Config.DEVICE)

    criterion = FocalLoss(alpha=1, gamma=2)
    scaler = GradScaler(enabled=Config.USE_AMP)
    history = TrainingHistory()
    early_stopping = EarlyStopping()
    best_val_f1 = 0.0
    best_epoch = 0

    print(f"\nTraining with {len(train_dataset)} samples, validating with {len(val_dataset)} samples, testing with {len(test_dataset)} samples")

    # Stage 1: Freeze Backbone, Train Classifier
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

    # Stage 2: Unfreeze All Layers, Fine-Tune
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

    # Test Evaluation
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load("best_rsna_model_v2.pth"))
    test_metrics = evaluate(model, test_loader, criterion, Config.DEVICE, "test")

    # Save Plots
    print("\nGenerating visualization plots...")
    plot_training_history(history)
    plot_confusion_matrix(val_metrics['confusion_matrix'])
    plot_roc_curve(val_metrics['all_labels'], val_metrics['all_probs'])

    # Save Results
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