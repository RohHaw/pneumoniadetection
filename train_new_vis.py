import os
from torchvision import datasets, transforms, models
import torch
from torch.utils.data import DataLoader, random_split, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

# Configuration
class Config:
    DATASET_DIR = "../archive_combined/"  # Directory containing all images
    IMAGE_SIZE = 224
    BATCH_SIZE = 64
    NUM_CLASSES = 2
    EPOCHS = 20
    BASE_LR = 0.001
    WEIGHT_DECAY = 0.0001
    DROPOUT = 0.3
    USE_AMP = True
    PATIENCE = 5
    MIN_DELTA = 0.001
    CLASS_WEIGHTS = torch.tensor([1.0, 2.0], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1

# Training history tracker
class TrainingHistory:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_f1s = []
        self.val_roc_aucs = []

# Visualization functions
def plot_training_history(history):
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(history.train_losses, label='Train Loss')
    plt.plot(history.val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(2, 2, 2)
    plt.plot(history.train_accuracies, label='Train Accuracy')
    plt.plot(history.val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot F1 scores
    plt.subplot(2, 2, 3)
    plt.plot(history.val_f1s, label='Validation F1')
    plt.title('Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    # Plot ROC-AUC scores
    plt.subplot(2, 2, 4)
    plt.plot(history.val_roc_aucs, label='Validation ROC-AUC')
    plt.title('Validation ROC-AUC Score')
    plt.xlabel('Epoch')
    plt.ylabel('ROC-AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_confusion_matrix(conf_matrix, class_names=['Class 0', 'Class 1']):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
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
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.close()

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomResizedCrop(Config.IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAutocontrast(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def train(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=Config.USE_AMP):
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

def evaluate(model, dataloader, criterion, device, phase="val"):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            with autocast(enabled=Config.USE_AMP):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                probs = torch.softmax(outputs, dim=1)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
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

def main():
    # Load and split dataset
    print("Loading dataset...")
    full_dataset = datasets.ImageFolder(Config.DATASET_DIR)
    total_size = len(full_dataset)
    train_size = int(Config.TRAIN_SPLIT * total_size)
    val_size = int(Config.VAL_SPLIT * total_size)
    test_size = total_size - train_size - val_size

    # Create train, validation, and test splits
    train_subset, val_subset, test_subset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create custom datasets with appropriate transforms
    train_dataset = CustomDataset(train_subset, train_transform)
    val_dataset = CustomDataset(val_subset, eval_transform)
    test_dataset = CustomDataset(test_subset, eval_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Model setup
    print("Setting up model...")
    model = models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Dropout(p=Config.DROPOUT),
        nn.Linear(model.fc.in_features, Config.NUM_CLASSES)
    )
    model = model.to(Config.DEVICE)

    criterion = nn.CrossEntropyLoss(weight=Config.CLASS_WEIGHTS)
    optimizer = optim.AdamW(model.parameters(), lr=Config.BASE_LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    scaler = GradScaler(enabled=Config.USE_AMP)

    # Initialize training history tracker
    history = TrainingHistory()
    early_stopping = EarlyStopping()
    best_val_f1 = 0.0
    best_epoch = 0

    print(f"\nTraining with {len(train_dataset)} samples, validating with {len(val_dataset)} samples, testing with {len(test_dataset)} samples")

    # Training loop
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        
        # Train
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, Config.DEVICE, scaler)
        print(f"\nTrain Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, Config.DEVICE, "val")
        print(f"Validation Loss: {val_metrics['loss']:.4f}")
        print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Validation F1: {val_metrics['f1']:.4f}")
        print(f"Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Record history
        history.train_losses.append(train_loss)
        history.train_accuracies.append(train_acc)
        history.val_losses.append(val_metrics['loss'])
        history.val_accuracies.append(val_metrics['accuracy'])
        history.val_f1s.append(val_metrics['f1'])
        history.val_roc_aucs.append(val_metrics['roc_auc'])
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            torch.save(model.state_dict(), "best_model_split_vis.pth")
            print("New best model saved!")
        
        # Early stopping
        early_stopping(val_metrics['loss'])
        if early_stopping.early_stop:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

    # Load best model and evaluate on test set
    print("\nEvaluating best model on test set...")
    model.load_state_dict(torch.load("best_model_split_vis.pth"))
    test_metrics = evaluate(model, test_loader, criterion, Config.DEVICE, "test")

    # Generate and save plots
    print("\nGenerating visualization plots...")
    plot_training_history(history)
    plot_confusion_matrix(test_metrics['confusion_matrix'])
    plot_roc_curve(test_metrics['all_labels'], test_metrics['all_probs'])
    print("Plots saved as training_history.png, confusion_matrix.png, and roc_curve.png")

    # Save results
    with open('model_results_split_vis.txt', 'w') as f:
        f.write("Final Model Results\n")
        f.write("==================\n\n")
        f.write(f"Dataset Split:\n")
        f.write(f"Train samples: {len(train_dataset)}\n")
        f.write(f"Validation samples: {len(val_dataset)}\n")
        f.write(f"Test samples: {len(test_dataset)}\n\n")
        f.write(f"Best Validation F1: {best_val_f1:.4f} (Epoch {best_epoch+1})\n\n")
        f.write("Test Set Metrics:\n")
        f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Recall: {test_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {test_metrics['f1']:.4f}\n")
        f.write(f"ROC-AUC: {test_metrics['roc_auc']:.4f}\n")

    print("\nTraining completed! Results saved to model_results_split_vis.txt")

if __name__ == "__main__":
    main()