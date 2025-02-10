import os
import wandb
from torchvision import datasets, transforms, models
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

# Initialize wandb
wandb.init(project="pneumonia-classification", config={
    "epochs": 20,
    "batch_size": 64,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "dropout": 0.3,
    "image_size": 224,
    "use_amp": True,
    "patience": 5,
    "min_delta": 0.001,
    "class_weights": [1.0, 2.0]  # Adjust based on dataset imbalance
})

# Configuration
class Config:
    DATASET_DIR = "archive/chest_xray"
    IMAGE_SIZE = wandb.config.image_size
    BATCH_SIZE = wandb.config.batch_size
    NUM_CLASSES = 2
    EPOCHS = wandb.config.epochs
    BASE_LR = wandb.config.learning_rate
    WEIGHT_DECAY = wandb.config.weight_decay
    DROPOUT = wandb.config.dropout
    USE_AMP = wandb.config.use_amp
    PATIENCE = wandb.config.patience
    MIN_DELTA = wandb.config.min_delta
    CLASS_WEIGHTS = torch.tensor(wandb.config.class_weights, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enhanced data augmentation
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

# Load datasets
train_dataset = datasets.ImageFolder(os.path.join(Config.DATASET_DIR, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(Config.DATASET_DIR, "val"), transform=eval_transform)
test_dataset = datasets.ImageFolder(os.path.join(Config.DATASET_DIR, "test"), transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Model setup
model = models.resnet50(pretrained=True)
model.fc = nn.Sequential(
    nn.Dropout(p=Config.DROPOUT),
    nn.Linear(model.fc.in_features, Config.NUM_CLASSES)
)
model = model.to(Config.DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=Config.CLASS_WEIGHTS)
optimizer = optim.AdamW(model.parameters(), lr=Config.BASE_LR, weight_decay=Config.WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
scaler = GradScaler(enabled=Config.USE_AMP)

# Training and evaluation functions
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
            wandb.log({"Batch Loss": loss.item()})
    
    accuracy = correct / total
    return running_loss / len(dataloader), accuracy

def evaluate(model, dataloader, criterion, device, phase="val"):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
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
early_stopping = EarlyStopping()
best_val_f1 = 0.0
best_epoch = 0

for epoch in range(Config.EPOCHS):
    print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
    
    # Train
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, Config.DEVICE, scaler)
    wandb.log({"Train Loss": train_loss, "Train Accuracy": train_acc})
    
    # Validate
    val_metrics = evaluate(model, val_loader, criterion, Config.DEVICE, "val")
    wandb.log({
        "Validation Loss": val_metrics['loss'],
        "Validation Accuracy": val_metrics['accuracy'],
        "Validation F1": val_metrics['f1'],
        "Validation ROC-AUC": val_metrics['roc_auc']
    })
    
    # Update learning rate
    scheduler.step()
    
    # Save best model
    if val_metrics['f1'] > best_val_f1:
        best_val_f1 = val_metrics['f1']
        best_epoch = epoch
        torch.save(model.state_dict(), "best_model_new.pth")
        print("New best model saved!")
    
    # Early stopping
    early_stopping(val_metrics['loss'])
    if early_stopping.early_stop:
        print(f"Early stopping triggered after epoch {epoch+1}")
        break

# Load best model and evaluate on test set
model.load_state_dict(torch.load("best_model_new.pth"))
test_metrics = evaluate(model, test_loader, criterion, Config.DEVICE, "test")

# Log test metrics
wandb.log({
    "Test Accuracy": test_metrics['accuracy'],
    "Test F1": test_metrics['f1'],
    "Test ROC-AUC": test_metrics['roc_auc']
})

# Save results
with open('model_new_results.txt', 'w') as f:
    f.write("Final Model Results\n")
    f.write("==================\n\n")
    f.write(f"Best Validation F1: {best_val_f1:.4f} (Epoch {best_epoch+1})\n\n")
    f.write("Test Set Metrics:\n")
    f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
    f.write(f"Precision: {test_metrics['precision']:.4f}\n")
    f.write(f"Recall: {test_metrics['recall']:.4f}\n")
    f.write(f"F1 Score: {test_metrics['f1']:.4f}\n")
    f.write(f"ROC-AUC: {test_metrics['roc_auc']:.4f}\n")

# Finish wandb run
wandb.finish()