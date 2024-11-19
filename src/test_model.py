import torch
import torch.nn as nn
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

def test_noise_robustness():
    """Test model's robustness to input noise"""
    model = MNISTModel()
    latest_model = get_latest_model()
    model.load_state_dict(torch.load(latest_model))
    model.eval()

    # Load a batch of test images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root='./data', 
                                train=False, 
                                transform=transform,
                                download=True)
    
    # Test first 100 images
    correct_normal = 0
    correct_noisy = 0
    total = 100

    with torch.no_grad():
        for i in range(total):
            image, label = test_dataset[i]
            image = image.unsqueeze(0)  # Add batch dimension

            # Test normal image
            output = model(image)
            pred = output.argmax(dim=1)
            correct_normal += (pred == label).item()

            # Add noise to image
            noise = torch.randn_like(image) * 0.1  # 10% Gaussian noise
            noisy_image = image + noise
            
            # Test noisy image
            output_noisy = model(noisy_image)
            pred_noisy = output_noisy.argmax(dim=1)
            correct_noisy += (pred_noisy == label).item()

    normal_accuracy = 100 * correct_normal / total
    noisy_accuracy = 100 * correct_noisy / total
    
    # Noisy accuracy should not be more than 10% worse than normal accuracy
    assert noisy_accuracy >= normal_accuracy - 10, \
        f"Model not robust to noise. Normal: {normal_accuracy}%, Noisy: {noisy_accuracy}%"

def test_gradient_health():
    """Test if model gradients are healthy (not vanishing/exploding)"""
    model = MNISTModel()
    criterion = nn.CrossEntropyLoss()
    
    # Create a small batch of test data
    test_input = torch.randn(32, 1, 28, 28)  # batch of 32 images
    test_target = torch.randint(0, 10, (32,))  # random targets
    
    # Enable gradient tracking
    test_input.requires_grad = True
    
    # Forward pass
    output = model(test_input)
    loss = criterion(output, test_target)
    
    # Backward pass
    loss.backward()
    
    # Check gradients for each layer
    healthy_gradients = True
    gradient_info = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            gradient_info[name] = grad_norm
            
            # Check if gradients are too small (vanishing) or too large (exploding)
            if grad_norm < 1e-5:
                healthy_gradients = False
                print(f"Warning: Possibly vanishing gradient in {name}: {grad_norm}")
            elif grad_norm > 1e2:
                healthy_gradients = False
                print(f"Warning: Possibly exploding gradient in {name}: {grad_norm}")
    
    # Assert gradients are healthy
    assert healthy_gradients, "Unhealthy gradients detected! Check gradient norms in test output."
    
    # Additional check: Verify gradients exist for all parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for parameter: {name}"

def test_class_wise_accuracy():
    """Test if model performs well across all digit classes"""
    model = MNISTModel()
    latest_model = get_latest_model()
    model.load_state_dict(torch.load(latest_model))
    model.eval()

    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root='./data', 
                                train=False, 
                                transform=transform,
                                download=True)
    
    # Initialize counters for each class
    class_correct = {i: 0 for i in range(10)}
    class_total = {i: 0 for i in range(10)}

    with torch.no_grad():
        for i in range(len(test_dataset)):
            image, label = test_dataset[i]
            output = model(image.unsqueeze(0))
            pred = output.argmax(dim=1).item()
            
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1

    # Calculate accuracy for each class
    class_accuracies = {}
    for digit in range(10):
        accuracy = (class_correct[digit] / class_total[digit]) * 100
        class_accuracies[digit] = accuracy
        
        # Assert each class has at least 90% accuracy
        assert accuracy >= 90, f"Poor performance on digit {digit}: {accuracy:.2f}%"
        
        # Print accuracies for visibility
        print(f"Digit {digit} Accuracy: {accuracy:.2f}%")

    # Check if accuracies are balanced (max difference < 10%)
    max_acc = max(class_accuracies.values())
    min_acc = min(class_accuracies.values())
    acc_difference = max_acc - min_acc
    
    assert acc_difference < 10, \
        f"Unbalanced performance across classes. Difference: {acc_difference:.2f}%"

def test_prediction_confidence():
    """Test if model's confidence aligns with its predictions"""
    model = MNISTModel()
    latest_model = get_latest_model()
    model.load_state_dict(torch.load(latest_model))
    model.eval()

    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root='./data', 
                                train=False, 
                                transform=transform,
                                download=True)
    
    # Test first 100 images
    correct_confidences = []
    incorrect_confidences = []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for i in range(100):
            image, label = test_dataset[i]
            output = model(image.unsqueeze(0))
            
            # Get prediction and confidence
            probabilities = softmax(output)
            confidence, pred = torch.max(probabilities, dim=1)
            confidence = confidence.item()
            pred = pred.item()
            
            # Store confidence based on correctness
            if pred == label:
                correct_confidences.append(confidence)
            else:
                incorrect_confidences.append(confidence)

    # Calculate average confidences
    avg_correct_conf = sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0
    avg_incorrect_conf = sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else 1

    # Print for visibility
    print(f"Average confidence when correct: {avg_correct_conf:.3f}")
    if incorrect_confidences:
        print(f"Average confidence when incorrect: {avg_incorrect_conf:.3f}")

    # Assertions
    assert avg_correct_conf > 0.9, \
        f"Model not confident enough in correct predictions: {avg_correct_conf:.3f}"
    
    if incorrect_confidences:
        assert avg_incorrect_conf < 0.8, \
            f"Model too confident in incorrect predictions: {avg_incorrect_conf:.3f}"

if __name__ == "__main__":
    pytest.main([__file__]) 