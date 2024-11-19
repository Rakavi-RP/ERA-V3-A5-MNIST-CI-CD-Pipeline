import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np

def show_augmented_samples():
    # Create directory for sample images if it doesn't exist
    import os
    if not os.path.exists('sample_images'):
        os.makedirs('sample_images')

    # Basic transform for original images
    basic_transform = transforms.ToTensor()

    # Increased rotation angle for more visible effect
    rotation_transform = transforms.RandomRotation(degrees=30)

    # Load dataset
    dataset = MNIST(root='./data', train=True, download=True, transform=basic_transform)

    # Create figure with 2×5 grid and extra vertical space
    fig = plt.figure(figsize=(15, 8))  # Increased height
    gs = fig.add_gridspec(3, 5, height_ratios=[0.2, 1, 1])  # 3 rows, top row for title
    
    # Create axes for the images
    axes = []
    for row in range(1, 3):  # Skip the first row (reserved for title)
        row_axes = []
        for col in range(5):
            ax = fig.add_subplot(gs[row, col])
            row_axes.append(ax)
        axes.append(row_axes)

    # Add row headers as subplot titles - adjusted y-coordinates
    fig.text(0.5, 0.95, 'ORIGINAL IMAGES', ha='center', va='center', fontsize=16, fontweight='bold')  # Moved from 0.55 to 0.95
    fig.text(0.5, 0.45, 'AUGMENTED IMAGES', ha='center', va='center', fontsize=16, fontweight='bold')

    # Explicitly choosing one of each digit for better visualization
    selected_indices = []
    digits_seen = set()
    
    # Find one example of each digit (0-9)
    for idx, (_, label) in enumerate(dataset):
        if label not in digits_seen and len(selected_indices) < 5:
            selected_indices.append(idx)
            digits_seen.add(label)
    
    for idx, sample_idx in enumerate(selected_indices):
        image, label = dataset[sample_idx]
        
        # Original image
        axes[0][idx].imshow(image.squeeze(), cmap='gray')
        axes[0][idx].set_title(f'Digit: {label}')
        axes[0][idx].axis('off')
        
        # Rotated image
        rotated_image = rotation_transform(image)
        axes[1][idx].imshow(rotated_image.squeeze(), cmap='gray')
        axes[1][idx].axis('off')

    plt.tight_layout()
    plt.savefig('sample_images/augmentation_samples.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    show_augmented_samples() 