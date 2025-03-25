

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import random


class ChestXrayValidationDataset(Dataset):
    def __init__(self, xray_dirs, non_xray_dir, transform=None):
        """
        Args:
            xray_dirs (list): List of directories containing chest X-ray images
            non_xray_dir (str): Directory containing non-X-ray images
            transform: Optional transform to be applied on images
        """
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load chest X-rays (positive examples)
        for xray_dir in xray_dirs:
            for root, _, files in os.walk(xray_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(root, file))
                        self.labels.append(1)  # 1 for chest X-ray
                        
        # Load non-X-ray images (negative examples)
        non_xray_files = []
        for root, _, files in os.walk(non_xray_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    non_xray_files.append(os.path.join(root, file))
        
        # Randomly sample same number of non-X-ray images as X-rays
        selected_non_xrays = random.sample(non_xray_files, min(len(self.labels), len(non_xray_files)))
        self.images.extend(selected_non_xrays)
        self.labels.extend([0] * len(selected_non_xrays))  # 0 for non-X-ray

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
            # Return a default item in case of error
            return torch.zeros((3, 224, 224)), self.labels[idx]

class ChestXrayValidator:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, 2)
        )
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def train(self, xray_dirs, non_xray_dir, epochs=10, batch_size=32, save_path='Training/xray_validator.pth'):
        """Train the validation model"""
        # Prepare dataset
        dataset = ChestXrayValidationDataset(xray_dirs, non_xray_dir, self.transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        best_val_acc = 0
        self.model.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Training phase
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
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f'Model saved with validation accuracy: {val_acc:.2f}%')
            
            self.model.train()

    def validate_image(self, image):
        """
        Validate if an image is a chest X-ray
        Returns: bool, confidence
        """
        self.model.eval()
        with torch.no_grad():
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            is_xray = probabilities[0][1].item() > 0.5
            confidence = probabilities[0][1].item() * 100
            
            return is_xray, confidence