import os
import pandas as pd
import pydicom
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class RSNADataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        self.labels = pd.read_csv(labels_file)
        # Group by patientId to get unique images
        self.labels = self.labels.groupby("patientId").first().reset_index()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        patient_id = self.labels.iloc[idx]["patientId"]
        image_path = os.path.join(self.image_dir, f"{patient_id}.dcm")
        
        # Load DICOM image
        ds = pydicom.dcmread(image_path)
        image = Image.fromarray(ds.pixel_array).convert("RGB")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label (0=Normal, 1=Pneumonia)
        label = self.labels.iloc[idx]["Target"]
        
        return image, label, patient_id

# Define transform (same as in PneumoniaClassifier)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])