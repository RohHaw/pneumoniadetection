import os
from torchvision import datasets, transforms, models
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configuration
class Config:
    # Dataset parameters
    DATASET_DIR = "archive/chest_xray"
    IMAGE_SIZE = 224
    
    # Training parameters
    BATCH_SIZE = 64  # Increased from 32 to better utilize GPU
    NUM_CLASSES = 2
    EPOCHS = 20  # Set to 20 with early stopping
    BASE_LR = 0.001
    WEIGHT_DECAY = 0.0001
    
    # Early stopping parameters
    PATIENCE = 5  # Number of epochs to wait before early stopping
    MIN_DELTA = 0.001  # Minimum change to qualify as an improvement
    
    # Mixed precision training
    USE_AMP = True  # Use automatic mixed precision
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enhanced data augmentation
train_transform = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomAutocontrast(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation/Test transforms (no augmentation)
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

# Load datasets with different transforms for train and eval
train_dataset = datasets.ImageFolder(os.path.join(Config.DATASET_DIR, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(Config.DATASET_DIR, "val"), transform=eval_transform)
test_dataset = datasets.ImageFolder(os.path.join(Config.DATASET_DIR, "test"), transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
print(f"Class labels: {train_dataset.classes}")

# Model setup
model = models.resnet50(pretrained=True)
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.fc.in_features, Config.NUM_CLASSES)
)
model = model.to(Config.DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=Config.BASE_LR, weight_decay=Config.WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
scaler = GradScaler(enabled=Config.USE_AMP)

def train(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Use automatic mixed precision
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
            print(f"Batch {i}/{len(dataloader)} - Loss: {loss.item():.4f}")
    
    accuracy = correct / total
    return running_loss / len(dataloader), accuracy

def evaluate(model, dataloader, criterion, device, phase="val"):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            with autocast(enabled=Config.USE_AMP):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    accuracy = (all_predictions == all_labels).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    return {
        'loss': running_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }

# Training loop with early stopping
early_stopping = EarlyStopping()
best_val_acc = 0.0
best_epoch = 0

for epoch in range(Config.EPOCHS):
    print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
    
    # Train
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, Config.DEVICE, scaler)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    
    # Validate
    val_metrics = evaluate(model, val_loader, criterion, Config.DEVICE, "val")
    print(f"Validation Metrics:")
    print(f"Loss: {val_metrics['loss']:.4f}")
    print(f"Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"F1 Score: {val_metrics['f1']:.4f}")
    
    # Update learning rate
    scheduler.step(val_metrics['loss'])
    
    # Save best model
    if val_metrics['accuracy'] > best_val_acc:
        best_val_acc = val_metrics['accuracy']
        best_epoch = epoch
        torch.save(model.state_dict(), "best_model.pth")
        print("New best model saved!")
    
    # Early stopping
    early_stopping(val_metrics['loss'])
    if early_stopping.early_stop:
        print(f"Early stopping triggered after epoch {epoch+1}")
        break

print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}")

# Load best model and evaluate on test set
model.load_state_dict(torch.load("best_model.pth"))
test_metrics = evaluate(model, test_loader, criterion, Config.DEVICE, "test")

print("\nFinal Test Set Metrics:")
print(f"Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Precision: {test_metrics['precision']:.4f}")
print(f"Recall: {test_metrics['recall']:.4f}")
print(f"F1 Score: {test_metrics['f1']:.4f}")

# Save results
with open('model_results.txt', 'w') as f:
    f.write("Final Model Results\n")
    f.write("==================\n\n")
    f.write(f"Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch+1})\n\n")
    f.write("Test Set Metrics:\n")
    f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
    f.write(f"Precision: {test_metrics['precision']:.4f}\n")
    f.write(f"Recall: {test_metrics['recall']:.4f}\n")
    f.write(f"F1 Score: {test_metrics['f1']:.4f}\n")