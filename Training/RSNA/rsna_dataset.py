"""
Dataset module for loading RSNA Pneumonia Detection Challenge data.

This module defines a PyTorch Dataset class for loading and preprocessing chest X-ray images
from the RSNA Pneumonia Detection Challenge. It handles DICOM images, applies transformations,
and provides labels for pneumonia classification. The dataset is designed to work with the
PneumoniaClassifier model, ensuring consistent preprocessing.

Author: Rohman Hawrylak
Date: April 2025
"""

import os
import pandas as pd
import pydicom
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class RSNADataset(Dataset):
    """
    PyTorch Dataset for the RSNA Pneumonia Detection Challenge.

    Loads chest X-ray images in DICOM format from a specified directory and their corresponding
    labels from a CSV file. Groups images by patient ID to ensure unique samples and applies
    optional transformations for preprocessing. Designed for use with the PneumoniaClassifier
    model for binary classification (Normal vs. Pneumonia).

    Attributes:
        image_dir (str): Directory containing DICOM image files.
        labels (pandas.DataFrame): DataFrame with patient IDs and labels.
        transform (callable, optional): Transformations to apply to images.
    """
    def __init__(self, image_dir, labels_file, transform=None):
        """
        Initialise the RSNADataset with image directory, labels file, and optional transform.

        Args:
            image_dir (str): Path to the directory containing DICOM image files.
            labels_file (str): Path to the CSV file containing patient IDs and labels.
            transform (callable, optional): Optional transform to apply to images.
        """
        self.image_dir = image_dir
        # Load labels and group by patient ID to ensure unique images
        self.labels = pd.read_csv(labels_file)
        self.labels = self.labels.groupby("patientId").first().reset_index()
        self.transform = transform

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of unique patient images in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieve an image and its label by index.

        Loads a DICOM image for the specified patient, converts it to RGB, applies any
        specified transformations, and returns the image tensor and its corresponding label.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label)
                - image (torch.Tensor): Preprocessed image tensor.
                - label (int): Binary label (0=Normal, 1=Pneumonia).
        """
        # Get patient ID and construct image path
        patient_id = self.labels.iloc[idx]["patientId"]
        image_path = os.path.join(self.image_dir, f"{patient_id}.dcm")
        
        # Load DICOM image and convert to RGB
        ds = pydicom.dcmread(image_path)
        image = Image.fromarray(ds.pixel_array).convert("RGB")
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        
        # Get binary label (0=Normal, 1=Pneumonia)
        label = self.labels.iloc[idx]["Target"]
        
        return image, label

# Define transform for preprocessing (consistent with PneumoniaClassifier)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 pixels
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalise with ImageNet stats
])