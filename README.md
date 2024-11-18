# MNIST Classification with CI/CD Pipeline

This project implements a simple Convolutional Neural Network (CNN) for MNIST digit classification with an automated CI/CD pipeline using GitHub Actions.

## Project Structure 
```
.
├── .github
│ └── workflows
│ └── ml-pipeline.yml
├── src
│ ├── model.py # CNN model architecture
│ ├── train.py # Training script
│ └── test_model.py # Testing and validation
├── models/ # Saved model artifacts
├── .gitignore
└── requirements.txt


## Model Architecture


- 2 Convolutional layers with ReLU activation and max pooling
- 2 Fully connected layers
- Less than 25,000 parameters
- Input: 28x28 grayscale images
- Output: 10 classes (digits 0-9)

## Features

- Automated model training
- Model validation checks:
  - Parameter count < 25,000
  - Input shape compatibility (28x28)
  - Output shape verification (10 classes)
  - Accuracy threshold > 95% on the test set after 1 epoch of training.
- CPU-only implementation for GitHub Actions compatibility
- Automated model artifact storage

## Setup and Running Locally

1. Create and activate virtual environment:

        python -m venv env
        source env/bin/activate # On Windows: env\Scripts\activate

2. Install dependencies:

        pip install -r requirements.txt

3. Train the model:

        python src/train.py

4. Run tests:

        pytest src/test_model.py -v


## CI/CD Pipeline

The GitHub Actions workflow automatically:
1. Sets up Python environment
2. Installs CPU-only PyTorch dependencies
3. Trains the model
4. Runs validation tests
5. Stores trained model artifacts

## Model Artifacts

Trained models are saved with timestamps in the format:

 mnist_model_YYYYMMDD_HHMMSS.pth


## Requirements

- Python 3.8+
- PyTorch (CPU version)
- pytest

## Notes

- Training is limited to 1 epoch 
- Model achieves >95% accuracy on test set
- All computations are CPU-based for CI/CD compatibility