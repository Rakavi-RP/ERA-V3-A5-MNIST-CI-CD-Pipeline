import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTModel
from datetime import datetime
import os

def train():
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Device configuration
    device = torch.device('cpu')

    # MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', 
                                 train=True, 
                                 transform=transform,
                                 download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=64,
                                             shuffle=True)

    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    total_step = len(train_loader)
    
    # Train for 1 epoch
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

    # Save the model with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    torch.save(model.state_dict(), f'models/mnist_model_{timestamp}.pth')
    
if __name__ == "__main__":
    train() 