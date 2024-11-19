import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime
import os

def show_augmented_samples():
    random.seed(datetime.now().timestamp())
    
    if not os.path.exists('sample_images'):
        os.makedirs('sample_images')

    # Basic transform for original images
    basic_transform = transforms.ToTensor()

    # Three mild augmentations (without ToTensor since image is already tensor)
    augmentation_transform = transforms.Compose([
        transforms.RandomRotation(degrees=15),      # Mild rotation
        transforms.RandomAffine(                    # Small shifts and scaling
            degrees=0,
            translate=(0.1, 0.1),                  # 10% shift
            scale=(0.9, 1.1)                       # 10% scale change
        )
    ])  # Removed ToTensor() from here

    dataset = MNIST(root='./data', train=True, download=True, transform=basic_transform)

    # Create figure with 2×5 grid
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(3, 5, height_ratios=[0.2, 1, 1])
    
    axes = []
    for row in range(1, 3):
        row_axes = []
        for col in range(5):
            ax = fig.add_subplot(gs[row, col])
            row_axes.append(ax)
        axes.append(row_axes)

    fig.text(0.5, 0.95, 'ORIGINAL IMAGES', ha='center', va='center', fontsize=16, fontweight='bold')
    fig.text(0.5, 0.45, 'AUGMENTED IMAGES', ha='center', va='center', fontsize=16, fontweight='bold')

    selected_indices = random.sample(range(len(dataset)), 5)
    
    for idx, sample_idx in enumerate(selected_indices):
        image, label = dataset[sample_idx]
        
        # Convert tensor to PIL Image for augmentation
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(image)
        
        # Original image
        axes[0][idx].imshow(image.squeeze(), cmap='gray')
        axes[0][idx].set_title(f'Digit: {label}')
        axes[0][idx].axis('off')
        
        # Augmented image - apply transforms to PIL Image then convert back to tensor
        aug_image = augmentation_transform(pil_image)
        aug_tensor = transforms.ToTensor()(aug_image)
        axes[1][idx].imshow(aug_tensor.squeeze(), cmap='gray')
        axes[1][idx].axis('off')

    plt.tight_layout()
    plt.savefig('sample_images/augmentation_samples.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    show_augmented_samples() 