import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from classifier import PneumoniaClassifier
from rsna_dataset import RSNADataset, transform
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Paths
RSNA_DATA_DIR = "/vol/scratch/SoC/misc/2024/sc21r2h/rsna_data"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "Training/best_model_final.pth")
NEW_MODEL_PATH = os.path.join(PROJECT_DIR, "best_model_final_finetuned.pth")

# Best hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
DROPOUT = 0.5  # Ensure this matches PneumoniaClassifier
CLASS_WEIGHTS = torch.tensor([1.0, 1.5], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 5

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
    plt.savefig(os.path.join(PROJECT_DIR, 'finetune_training_history.png'))
    plt.close()

def plot_confusion_matrix(conf_matrix, class_names=['Normal', 'Pneumonia']):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, 'finetune_confusion_matrix.png'))
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
    plt.savefig(os.path.join(PROJECT_DIR, 'finetune_roc_curve.png'))
    plt.close()

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
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

def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, device=DEVICE):
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.to(device)
    
    history = TrainingHistory()
    best_val_f1 = 0.0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation phase
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Record history
        history.train_losses.append(train_loss)
        history.train_accuracies.append(train_acc)
        history.val_losses.append(val_metrics['loss'])
        history.val_accuracies.append(val_metrics['accuracy'] * 100)  # Convert to percentage
        history.val_f1s.append(val_metrics['f1'])
        history.val_roc_aucs.append(val_metrics['roc_auc'])
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']*100:.2f}%")
        print(f"  Val F1: {val_metrics['f1']:.4f}, Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
        
        # Save best model based on val_f1
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            torch.save(model.model.state_dict(), NEW_MODEL_PATH)
            print("New best model saved!")
    
    # Generate and save plots
    print("\nGenerating visualization plots...")
    plot_training_history(history)
    plot_confusion_matrix(val_metrics['confusion_matrix'])
    plot_roc_curve(val_metrics['all_labels'], val_metrics['all_probs'])
    print("Plots saved as finetune_training_history.png, finetune_confusion_matrix.png, and finetune_roc_curve.png")

    # Save results to a text file
    with open(os.path.join(PROJECT_DIR, 'finetune_results.txt'), 'w') as f:
        f.write("Fine-Tuning Results\n")
        f.write("==================\n\n")
        f.write(f"Dataset Split:\n")
        f.write(f"Train samples: {len(train_loader.dataset)}\n")
        f.write(f"Validation samples: {len(val_loader.dataset)}\n\n")
        f.write(f"Best Validation F1: {best_val_f1:.4f} (Epoch {best_epoch+1})\n\n")
        f.write("Final Validation Metrics:\n")
        f.write(f"Accuracy: {val_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {val_metrics['precision']:.4f}\n")
        f.write(f"Recall: {val_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {val_metrics['f1']:.4f}\n")
        f.write(f"ROC-AUC: {val_metrics['roc_auc']:.4f}\n")

    print("\nFine-tuning completed! Results saved to finetune_results.txt")
    
    return model

def main():
    # Load dataset
    image_dir = os.path.join(RSNA_DATA_DIR, "train_images")
    labels_file = os.path.join(RSNA_DATA_DIR, "labels", "stage_2_train_labels.csv")
    
    dataset = RSNADataset(image_dir, labels_file, transform=transform)
    
    # Split into train and validation (80-20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Load model
    model = PneumoniaClassifier(model_path=MODEL_PATH, dropout=DROPOUT)
    
    # Fine-tune
    model = train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, device=DEVICE)

if __name__ == "__main__":
    main()