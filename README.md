# Visual and Textual Explainable AI for Pneumonia Detection

## Overview

This project develops an Explainable Artificial Intelligence (XAI) system for detecting pneumonia in chest X-ray images, enhancing clinical trust and diagnostic transparency. Submitted as part of a BSc Computer Science (Digital & Technology Solutions) dissertation at the University of Leeds (2024/25), the system integrates deep learning, visual explanations, textual insights, and uncertainty quantification to support clinicians in making informed decisions. Pneumonia, a leading cause of global mortality with over 2 million deaths annually, poses diagnostic challenges due to subtle radiographic signs. This project addresses these by combining a fine-tuned ResNet-50 model, Grad-CAM visualisations, Monte Carlo Dropout for uncertainty estimation, and a Gemini API-based clinical assistant for interactive textual explanations.

The system achieves high diagnostic accuracy (up to 96.59% on the Kaggle dataset), provides interpretable outputs via heatmaps and plain-language explanations, and supports real-time clinical use with an average inference time of 1.052 seconds. An interactive Streamlit interface enables clinicians to upload X-rays, review predictions, and query the system, making it a valuable tool for both clinical decision-making and medical education.

## Features

- **Input Validation**: A ResNet-18 model ensures inputs are valid chest X-rays, achieving 100% accuracy on a mixed dataset (Kaggle + ImageNet).
- **Pneumonia Detection**: A fine-tuned ResNet-50 model classifies X-rays as normal or pneumonia-affected, with robust performance on Kaggle (96.59% accuracy) and improved generalisation on RSNA (70.08% post-fine-tuning).
- **Visual Explanations**: Grad-CAM generates heatmaps highlighting regions influencing predictions, with an average IoU of 0.1379 on RSNA data.
- **Textual Explanations**: A Gemini API-based clinical assistant provides plain-language descriptions of predictions and heatmaps, enhancing clinician accessibility.
- **Uncertainty Quantification**: Monte Carlo Dropout (20 passes) estimates prediction variance, flagging ambiguous cases for further review (average variance 0.0028 on Kaggle).
- **User Interface**: A Streamlit-based interface allows users to upload images, view predictions, heatmaps, confidence intervals, and interact via a chat feature.
- **Real-Time Performance**: Processes images in ~1.052 seconds (preprocessing: 0.003s, classification: 0.037s, Grad-CAM: 0.010s) with low resource usage (200 MB GPU memory).

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support (recommended for performance)
- Git
- pip

### Dependencies
The project uses the following Python libraries:
- `torch` and `torchvision` for deep learning
- `numpy` and `pandas` for data handling
- `matplotlib` and `seaborn` for visualisation
- `streamlit` for the user interface
- `albumentations` for data augmentation
- `Pillow` for image processing
- `google-generativeai` for the Gemini API
- `scikit-learn` for metrics
- `tqdm` for progress tracking

### Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/RohHaw/pneumoniadetection.git
   cd pneumoniadetection
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```


4. **Configure Gemini API**:
   - Obtain an API key from [Google Cloud](https://cloud.google.com/gemini).
   - Set the key as an environment variable:
     ```bash
     export GEMINI_API_KEY='your-api-key'
     ```


## Usage

### Running the Application
1. **Launch the Streamlit Interface**:
   ```bash
   streamlit run main.py
   ```
   - Access the interface at `http://localhost:8501` in your browser.
   - Upload a chest X-ray image, view predictions, heatmaps, uncertainty estimates, and interact with the clinical assistant via the chat feature.

2. **Training the Model**:
   - Use `train_UCSD.py` to train the ResNet-50 model on the UCSD dataset:
     ```bash
     python train_UCSD.py
     ```
   - Adjust hyperparameters (e.g., epochs, learning rate) in the script as needed.

3. **Evaluating Performance**:
   - Run `test_on_rsna.py` for RSNA different dataset evaluation:
     ```bash
     python test_on_rsna.py
     ```
   - Use `evaluate_UCSD.py` for UCSD dataset evaluation:
     ```bash
     python evaluate_UCSD.py
     ```

4. **Grad-CAM Analysis**:
   - Generate and evaluate Grad-CAM heatmaps with `evaluate_gradcam.py`:
     ```bash
     python evaluate_gradcam.py
     ```

5. **Uncertainty Quantification**:
   - Perform Monte Carlo Dropout experiments with `run_mcd_experiment.py`:
     ```bash
     python run_mcd_experiment.py
     ```

6. **Performance Testing**:
   - Measure inference times and resource usage with `perf_testing.py`:
     ```bash
     python perf_testing.py
     ```

### Example Workflow
1. Upload a chest X-ray via the Streamlit interface.
2. The ResNet-18 validator confirms the image is a valid X-ray.
3. ResNet-50 predicts pneumonia probability (e.g., 95% pneumonia).
4. Grad-CAM generates a heatmap highlighting affected regions.
5. Monte Carlo Dropout provides a confidence interval (e.g., 95% ± 0.02).
6. The clinical assistant explains: "The model detected a high-intensity infiltrate in the upper lobe, consistent with pneumonia."
7. Query the assistant for clarification, e.g., "Why this region?" to receive detailed responses.

## Datasets

The project utilises two publicly available datasets:
- **Kaggle Chest X-Ray Images (Pneumonia)**:
  - Source: [Kaggle](https://www.kaggle.com/datasets/paultimothymoney/chest-xray-pneumonia)
  - Description: 5,856 pediatric chest X-rays (1,583 normal, 4,273 pneumonia).
  - Format: JPEG/PNG.
  - Usage: Training and testing ResNet-50, uncertainty quantification.
- **RSNA Pneumonia Detection Challenge**:
  - Source: [Kaggle](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
  - Description: 26,684 adult chest X-rays (20,672 normal, 6,012 pneumonia) with bounding box annotations.
  - Format: DICOM.
  - Usage: Fine-tuning ResNet-50, Grad-CAM evaluation.


## Results

### Diagnostic Accuracy
- **ResNet-18 Validator**: 100% accuracy, precision, recall, F1-score, and AUC-ROC on 1,759 images (879 Kaggle X-rays, 880 ImageNet non-X-rays).
- **ResNet-50 Classifier**:
  - **Kaggle (Original)**: Accuracy 96.59%, AUC-ROC 99.05%.
  - **RSNA (Original)**: Accuracy 40.18%, AUC-ROC 76.79% (poor due to domain shift).
  - **RSNA (Post-Fine-Tuning)**: Accuracy 70.08%, AUC-ROC 77.73%.
  - **Kaggle (Post-Fine-Tuning)**: Accuracy 77.65%, AUC-ROC 99.21% (trade-off from RSNA specialisation).

### Interpretability
- **Grad-CAM**: Average IoU 0.1379 on 200 RSNA pneumonia-positive samples (up from 0.035 pre-fine-tuning). 26% zero-IoU cases due to heatmap vs. bounding box mismatch, but high IoU (e.g., 0.6226) for confident predictions.

### Uncertainty Quantification
- **Monte Carlo Dropout**: Average variance 0.0028 on 390 Kaggle pneumonia-positive images. Higher uncertainty in mid-range probabilities (0.6-0.9), supporting clinical review of ambiguous cases.

### Usability
- Survey of 20 medical students (University of Leeds, March 2025) rated on a 5-point Likert scale:
  - Heatmap clarity: 4.65
  - Overall helpfulness: 4.50
  - Text explanations: 4.30-4.40
  - Clinical trust: 2.95 (lower due to skepticism among advanced students).

### Performance
- Total inference time: 1.052 seconds/image.
- Resource usage: 200 MB GPU memory, 0.58% CPU load, ensuring scalability.

## Project Structure

```
pneumoniadetection/
├── Training/               # Pre-trained model weights and training scripts
├── evaluation_images_and_data/ # Evaluation data and visualisations
├── evaluation/             # Evaluation scripts
├── classifier.py           # ResNet-50 classifier implementation
├── clinical_qa.py          # Gemini API-based clinical assistant
├── config.py               # Configuration constants for the application
├── gradcam.py              # Grad-CAM heatmap generation
├── main.py                 # Streamlit UI integration
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── ui_components           # UI code for streamlit
└── utils.py                # Preprocessing and utility functions
```

## Future Work

- **Technical Enhancements**:
  - Improve ResNet-50 generalisation using domain-adaptive training for accuracy >85% across datasets.
  - Enhance Grad-CAM precision (IoU >0.5) with attention mechanisms or multi-layer variants.
  - Expand uncertainty quantification with ensemble methods or Bayesian neural networks.
  - Optimise inference time to <0.5 seconds via model pruning or edge deployment.
  - Validate against radiologist diagnoses to boost clinical trust (>4.0).
- **Extensions**:
  - Extend to multi-disease detection (e.g., tuberculosis, lung cancer) using datasets like CheXpert.
  - Integrate with hospital PACS systems for real-time clinical use.
  - Deploy in low-resource settings with offline capability and simplified visuals.
  - Open-source with educational modules to support global medical training.

## Acknowledgements

- **Supervisor**: Dr. Sharib Ali for guidance and support.
- **Assessor**: Dr. Haiko Muller for valuable feedback.

## Contact

For questions or collaboration, contact Rohman Hawrylak via [sc21r2h@leeds.ac.uk](mailto:sc21r2h@leeds.ac.uk) or raise an issue on the GitHub repository.


---

*© 2024/25 The University of Leeds and Rohman Hawrylak*