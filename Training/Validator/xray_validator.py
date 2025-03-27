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
    def __init__(self, xray_dirs, non_xray_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        
        for xray_dir in xray_dirs:
            for root, _, files in os.walk(xray_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(root, file))
                        self.labels.append(1)
                        
        non_xray_files = []
        for root, _, files in os.walk(non_xray_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    non_xray_files.append(os.path.join(root, file))
        
        selected_non_xrays = random.sample(non_xray_files, min(len(self.labels), len(non_xray_files)))
        self.images.extend(selected_non_xrays)
        self.labels.extend([0] * len(selected_non_xrays))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return torch.zeros((3, 224, 224)), self.labels[idx]

class ChestXrayValidator:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, 2)
        )
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def train(self, xray_dirs, non_xray_dir, epochs=10, batch_size=32, save_path='Training/Validator/xray_validator2.pth'):
        # Prepare dataset with train/val/test split
        dataset = ChestXrayValidationDataset(xray_dirs, non_xray_dir, self.transform)
        train_size = int(0.7 * len(dataset))  # 70% train
        val_size = int(0.15 * len(dataset))  # 15% validation
        test_size = len(dataset) - train_size - val_size  # 15% test
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        best_val_acc = 0
        self.model.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            train_correct = 0
            train_total = 0
            
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
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f'Model saved with validation accuracy: {val_acc:.2f}%')
            
            self.model.train()
        
        # Test evaluation and visualization
        self.evaluate_and_visualize(test_loader, save_path)

    def evaluate_and_visualize(self, test_loader, save_path):
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
        
        # Confusion Matrix
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
        
        # ROC Curve
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
        self.model.eval()
        with torch.no_grad():
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            is_xray = probabilities[0][1].item() > 0.5
            confidence = probabilities[0][1].item() * 100
            return is_xray, confidence