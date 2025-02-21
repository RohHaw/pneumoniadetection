import os
from torchvision import datasets, transforms, models
import torch
from torch.utils.data import DataLoader, random_split, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR



# Configuration
class Config:
    DATASET_DIR = "archive_combined/chest_xray/"  # Directory containing all images
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
        'confusion_matrix': conf_matrix
    }

# Training loop
print(f"\nTraining with {len(train_dataset)} samples, validating with {len(val_dataset)} samples, testing with {len(test_dataset)} samples")

early_stopping = EarlyStopping()
best_val_f1 = 0.0
best_epoch = 0

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
    
    # Save best model
    if val_metrics['f1'] > best_val_f1:
        best_val_f1 = val_metrics['f1']
        best_epoch = epoch
        torch.save(model.state_dict(), "best_model.pth")
        print("New best model saved!")
    
    # Early stopping
    early_stopping(val_metrics['loss'])
    if early_stopping.early_stop:
        print(f"Early stopping triggered after epoch {epoch+1}")
        break

# Load best model and evaluate on test set
print("\nEvaluating best model on test set...")
model.load_state_dict(torch.load("best_model_split.pth"))
test_metrics = evaluate(model, test_loader, criterion, Config.DEVICE, "test")

# Save results
with open('model_results_split.txt', 'w') as f:
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

print("\nTraining completed! Results saved to model_results.txt")