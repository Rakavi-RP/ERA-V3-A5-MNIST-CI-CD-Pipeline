import torch
from torchvision import datasets, transforms
from model import MNISTModel
import pytest
import glob
import os

def get_latest_model():
    model_files = glob.glob('models/mnist_model_*.pth')
    if not model_files:
        raise FileNotFoundError("No model file found")
    return max(model_files)

def test_model_architecture():
    model = MNISTModel()
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Output shape is incorrect"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model has {total_params} parameters, should be less than 25000"

def test_model_accuracy():
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root='./data', 
                                train=False, 
                                transform=transform,
                                download=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=1000,
                                            shuffle=False)
    
    # Load the latest model
    model = MNISTModel()
    latest_model = get_latest_model()
    model.load_state_dict(torch.load(latest_model))
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 95, f"Model accuracy is {accuracy}%, should be > 95%"

if __name__ == "__main__":
    pytest.main([__file__]) 