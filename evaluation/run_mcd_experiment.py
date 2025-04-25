import os
import random
import torch
from PIL import Image
import pandas as pd
import numpy as np
from classifier import PneumoniaClassifier  # Your provided class

# Paths (adjust these to your local setup)
KAGGLE_PNEUMONIA_DIR = "/Users/rohmanhawrylak/Downloads/chest_xray/test/PNEUMONIA"
MODEL_PATH = "Training/UCSD/model_UCSD_finetuned_rsna.pth"  # Your model path
OUTPUT_CSV = "mcd_results_390.csv"

# Function to load random images
def load_random_images(directory, num_samples=390):
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

# Main experiment function
def run_mcd_experiment():
    # Initialize classifier
    classifier = PneumoniaClassifier(model_path=MODEL_PATH, mc_dropout_iterations=20, dropout=0.3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)
    print(f"Running on {device}")

    # Load 100 random pneumonia-positive images
    print(f"Loading 100 random pneumonia-positive X-rays from {KAGGLE_PNEUMONIA_DIR}...")
    image_data = load_random_images(KAGGLE_PNEUMONIA_DIR, num_samples=390)

    # Store results
    results = {
        "Image": [],
        "Mean_Pneumonia_Probability": [],
        "Variance_Pneumonia_Probability": []
    }

    # Process each image
    for idx, (filename, image) in enumerate(image_data, 1):
        print(f"Processing image {idx}/390: {filename}")
        try:
            # Run prediction with MCD
            prediction = classifier.predict(image)
            
            # Extract pneumonia probability and uncertainty
            mean_prob = prediction["probabilities"]["Pneumonia"] / 100.0  # Convert % to [0,1]
            std_prob = prediction["uncertainty"] / 100.0  # Max std from prediction, convert % to [0,1]
            variance = std_prob ** 2  # Variance = (standard deviation)^2

            # Store results
            results["Image"].append(filename)
            results["Mean_Pneumonia_Probability"].append(mean_prob)
            results["Variance_Pneumonia_Probability"].append(variance)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Run the experiment
    run_mcd_experiment()