"""
Script for training a chest X-ray validator model.

This module uses the ChestXrayValidator class to train a model for distinguishing chest X-ray
images (normal and pneumonia) from non-X-ray images. The script specifies directories for
X-ray and non-X-ray images, training hyperparameters, and a save path for the trained model.

Author: Rohman Hawrylak
Date: April 2025
"""

from xray_validator import ChestXrayValidator

def main():
    """
    Main function to train the chest X-ray validator model.

    Initialises the ChestXrayValidator, trains it on the specified X-ray and non-X-ray image
    directories with given hyperparameters, and saves the trained model to the specified path.
    """
    # Initialise the validator
    validator = ChestXrayValidator()

    # Train the model with specified directories and hyperparameters
    validator.train(
        xray_dirs=[
            'D:\\OneDrive\\OneDrive - University of Leeds\\Uni\\Year 3\\Dissertation\\xray_validator_dataset\\chest_xrays\\normal',
            'D:\\OneDrive\\OneDrive - University of Leeds\\Uni\\Year 3\\Dissertation\\xray_validator_dataset\\chest_xrays\\pneumonia'
        ],  # Directories containing normal and pneumonia chest X-ray images
        non_xray_dir='D:\\OneDrive\\OneDrive - University of Leeds\\Uni\\Year 3\\Dissertation\\xray_validator_dataset\\non_xrays',  # Directory containing non-X-ray images
        epochs=10,  # Number of training epochs
        batch_size=32,  # Batch size for training
        save_path='Training/Validator/xray_validator2.pth'  # Path to save the trained model
    )

if __name__ == "__main__":
    main()