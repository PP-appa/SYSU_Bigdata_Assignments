import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import time
import json
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware computing device selected: {device.type.upper()}")

    print("Preparing DataLoader...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    print("Initializing ResNet18 model...")
    model = resnet18(num_classes=10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 5
    print(f"Starting {device.type.upper()} training for {EPOCHS} epochs...")

    total_train_time = 0.0
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        for step, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % 100 == 0 and step > 0:
                print(f"  -> Epoch [{epoch+1}/{EPOCHS}], Step [{step}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_time = time.time() - start_time
        total_train_time += epoch_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s | Average Loss: {running_loss/len(train_loader):.4f}\n")

    print("Evaluation phase: Testing on 10,000 images...")
    model.eval()
    correct = 0
    total = 0
    start_test_time = time.time()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_time = time.time() - start_test_time
    accuracy = 100 * correct / total

    print(f"Prediction completed in {test_time:.2f}s")
    print(f"ResNet Validation Accuracy on CIFAR-10: {accuracy:.2f}%")

    result_file = './output/model_results.json'
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    
    results = {}
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                pass

    results['ResNet_GPU'] = {'accuracy': float(f"{accuracy:.2f}"), 'train_time': float(f"{total_train_time:.2f}")}

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {result_file}")

if __name__ == "__main__":
    main()
