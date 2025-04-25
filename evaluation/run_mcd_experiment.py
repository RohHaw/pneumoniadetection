"""
Module for running Monte Carlo Dropout experiments on pneumonia detection.

This script conducts an experiment to quantify prediction uncertainty using Monte Carlo Dropout
(MCD) on a set of pneumonia-positive chest X-ray images from the Kaggle dataset. It utilises the
PneumoniaClassifier to process images, compute mean pneumonia probabilities and their variances,
and save results to a CSV file. The script supports GPU or CPU execution and includes error handling
for robust processing.

Author: Rohman Hawrylak
Date: April 2025
"""

import os
import random
import torch
from PIL import Image
import pandas as pd
import numpy as np
from classifier import PneumoniaClassifier

# Define file paths for input data, model, and output
KAGGLE_PNEUMONIA_DIR = "/Users/rohmanhawrylak/Downloads/chest_xray/test/PNEUMONIA"
MODEL_PATH = "Training/UCSD/model_UCSD_finetuned_rsna.pth"
OUTPUT_CSV = "mcd_results_390.csv"

def load_random_images(directory, num_samples=390):
    """
    Load a random selection of images from a specified directory.

    Selects a specified number of image files with supported extensions, shuffles them, and loads
    each as an RGB PIL Image object.

    Args:
        directory (str): Path to the directory containing image files.
        num_samples (int, optional): Number of images to load. Defaults to 390.

    Returns:
        list: List of tuples, each containing the filename and corresponding PIL Image object.

    Raises:
        ValueError: If the directory contains fewer images than requested.
    """
    image_files = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if len(image_files) < num_samples:
        raise ValueError(f"Not enough images in {directory}. Found {len(image_files)}, need {num_samples}.")
    random.shuffle(image_files)
    selected_files = image_files[:num_samples]
    images = []
    for file in selected_files:
        img_path = os.path.join(directory, file)
        img = Image.open(img_path).convert('RGB')
        images.append((file, img))
    return images

def run_mcd_experiment():
    """
    Run a Monte Carlo Dropout experiment on pneumonia-positive chest X-rays.

    Loads 390 random pneumonia-positive images from the Kaggle dataset, processes them using the
    PneumoniaClassifier with Monte Carlo Dropout, and computes mean pneumonia probabilities and
    their variances. Results are saved to a CSV file for analysis.

    Returns:
        None: Outputs results to a CSV file and prints progress updates.
    """
    # Initialise the classifier with specified model and dropout settings
    classifier = PneumoniaClassifier(model_path=MODEL_PATH, mc_dropout_iterations=20, dropout=0.3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)
    print(f"Running on {device}")

    # Load random pneumonia-positive images
    print(f"Loading 100 random pneumonia-positive X-rays from {KAGGLE_PNEUMONIA_DIR}...")
    image_data = load_random_images(KAGGLE_PNEUMONIA_DIR, num_samples=390)

    # Initialise results dictionary
    results = {
        "Image": [],
        "Mean_Pneumonia_Probability": [],
        "Variance_Pneumonia_Probability": []
    }

    # Process each image
    for idx, (filename, image) in enumerate(image_data, 1):
        print(f"Processing image {idx}/390: {filename}")
        try:
            # Generate prediction with Monte Carlo Dropout
            prediction = classifier.predict(image)
            
            # Extract mean probability and uncertainty for pneumonia
            mean_prob = prediction["probabilities"]["Pneumonia"] / 100.0
            std_prob = prediction["uncertainty"] / 100.0
            variance = std_prob ** 2

            # Store results
            results["Image"].append(filename)
            results["Mean_Pneumonia_Probability"].append(mean_prob)
            results["Variance_Pneumonia_Probability"].append(variance)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    """
    Entry point for running the Monte Carlo Dropout experiment.

    Sets random seeds for reproducibility and executes the run_mcd_experiment function to process
    390 pneumonia-positive images.
    """
    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Run the experiment
    run_mcd_experiment()