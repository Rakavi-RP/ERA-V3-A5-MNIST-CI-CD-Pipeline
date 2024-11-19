# MNIST Classification with CI/CD Pipeline

This project implements a simple Convolutional Neural Network (CNN) for MNIST digit classification with an automated CI/CD pipeline using GitHub Actions.

## Project Structure
```
.
├── .github
│   └── workflows
│       └── ml-pipeline.yml    # CI/CD workflow configuration
├── src
│   ├── model.py              # CNN model architecture
│   ├── train.py              # Training script with augmentation
│   ├── show_augmentations.py # Augmentation visualization
│   └── test_model.py         # Testing and validation
├── models/                   # Saved model artifacts
├── sample_images/           # Augmented image samples
├── .gitignore
└── requirements.txt
```

## Model Architecture
- 2 Convolutional layers with ReLU activation and max pooling
- 2 Fully connected layers
- Less than 25,000 parameters
- Input: 28x28 grayscale images
- Output: 10 classes (digits 0-9)

## Features

### Data Augmentation
- Random rotation (±30 degrees)
- Visualization of original vs augmented images
- Augmentation samples saved during training

### Model Validation
Basic Tests:
- Parameter count < 25,000
- Input shape compatibility (28x28)
- Output shape verification (10 classes)
- Overall accuracy > 95%

Advanced Tests:
1. Noise Robustness Test
   - Adds 10% Gaussian noise
   - Ensures predictions remain stable
   - Accuracy drop < 10% with noise

2. Gradient Health Test
   - Checks for vanishing/exploding gradients
   - Ensures proper gradient flow
   - Validates training stability

3. Class-wise Accuracy Test
   - Tests each digit separately
   - Ensures >90% accuracy per digit
   - Checks balanced performance across classes

4. Prediction Confidence Test
   - Verifies model confidence
   - High confidence (>90%) for correct predictions
   - Lower confidence (<80%) for incorrect ones

## Setup and Running Locally

1. Create and activate virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Generate augmentation samples:
```bash
python src/show_augmentations.py
```

4. Train the model:
```bash
python src/train.py
```

5. Run tests:
```bash
python -m pytest src/test_model.py -v
```

## CI/CD Pipeline

The GitHub Actions workflow automatically:
1. Sets up Python environment
2. Installs CPU-only PyTorch dependencies
3. Generates augmentation samples
4. Trains the model
5. Runs all validation tests
6. Stores artifacts (model & images)

### Viewing Artifacts in GitHub Actions
1. Go to the Actions tab in your repository
2. Click on the latest workflow run
3. Scroll down to "Artifacts"
4. Download "model-and-samples" zip file
5. Extract the zip file to see:
   - `models/` folder containing trained model file (mnist_model_YYYYMMDD_HHMMSS.pth)
   - `sample_images/` folder containing augmentation_samples.png showing:
     * Top row: Original MNIST digits
     * Bottom row: Same digits with random rotation


## Model Artifacts

Trained models are saved with timestamps:
```
mnist_model_YYYYMMDD_HHMMSS.pth
```

## Requirements
- Python 3.8+
- PyTorch (CPU version)
- pytest
- matplotlib
- numpy<2.0
- pillow
- psutil

## Notes
- Training limited to 1 epoch
- All computations are CPU-based for CI/CD compatibility
- Augmentation helps improve model robustness
- Comprehensive testing ensures model quality