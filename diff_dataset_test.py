
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Simple Pneumonia Classifier for classification only
class SimplePneumoniaClassifier(nn.Module):
    def __init__(self, model_path="Training/UCSD/model_UCSD.pth", input_size=224):
        super(SimplePneumoniaClassifier, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(weights=None)
        # Match the training structure: Sequential with Dropout and Linear
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3),  # Matches training dropout
            nn.Linear(self.model.fc.in_features, 2)
        )

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.classes = ['Normal', 'Pneumonia']
        self.input_size = input_size

    def forward(self, x):
        return self.model(x)

    def predict(self, image):
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_class = np.argmax(probs)

        return {
            'class': self.classes[pred_class],
            'probabilities': {
                'Normal': probs[0] * 100,
                'Pneumonia': probs[1] * 100
            }
        }

# Evaluation function
def evaluate_model(classifier, dataloader):
    all_labels = []
    all_preds = []
    all_probs = []

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for images, labels in dataloader:
        for img, label in zip(images, labels):
            img_np = img.numpy().transpose(1, 2, 0)
            img_np = (img_np * std + mean) * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            result = classifier.predict(img_pil)
            pred_class = classifier.classes.index(result['class'])
            all_labels.append(label.item())
            all_preds.append(pred_class)
            all_probs.append(result['probabilities']['Pneumonia'] / 100)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    roc_auc = roc_auc_score(all_labels, all_probs)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix,
        'all_labels': all_labels,
        'all_probs': all_probs
    }

# Plotting functions
def plot_confusion_matrix(conf_matrix, class_names=['Normal', 'Pneumonia']):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_new_dataset.png')
    plt.close()

def plot_roc_curve(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(labels, probs):.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_new_dataset.png')
    plt.close()

# Main evaluation script
def main():
    data_dir = "C:/Users/rhawr/Documents/archive_combined"
    model_path = "Training/UCSD/model_UCSD.pth"
    output_file = "evaluation_results_new_dataset.txt"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading dataset...")
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Dataset size: {len(dataset)} images")
    print(f"Class distribution: {dict(zip(dataset.classes, [len([x for x in dataset.targets if x == i]) for i in range(len(dataset.classes))]))}")

    classifier = SimplePneumoniaClassifier(model_path=model_path)

    print("Evaluating model...")
    metrics = evaluate_model(classifier, dataloader)

    with open(output_file, 'w') as f:
        f.write("Evaluation Results on New Dataset\n")
        f.write("================================\n\n")
        f.write(f"Dataset Path: {data_dir}\n")
        f.write(f"Total Images: {len(dataset)}\n")
        f.write(f"Class Distribution: {dict(zip(dataset.classes, [len([x for x in dataset.targets if x == i]) for i in range(len(dataset.classes))]))}\n\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"ROC-AUC: {metrics['roc_auc']:.4f}\n")
        f.write(f"Confusion Matrix:\n{metrics['confusion_matrix'].tolist()}\n")

    print(f"Results saved to {output_file}")

    print("Generating visualizations...")
    plot_confusion_matrix(metrics['confusion_matrix'])
    plot_roc_curve(metrics['all_labels'], metrics['all_probs'])
    print("Visualizations saved as confusion_matrix_new_dataset.png, roc_curve_new_dataset.png")

if __name__ == "__main__":
    main()