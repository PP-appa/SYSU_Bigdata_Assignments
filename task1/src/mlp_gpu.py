import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import json
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device.type.upper()}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.flatten = nn.Flatten()
            self.network = nn.Sequential(
                nn.Linear(3072, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
            )

        def forward(self, x):
            x = self.flatten(x)
            return self.network(x)

    print("Initializing Multi-Layer Perceptron (MLP) model...")
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 10
    print(f"Training MLP for {EPOCHS} epochs...")

    total_train_time = 0.0
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_time = time.time() - start_time
        total_train_time += epoch_time
        print(f"Epoch {epoch+1}/{EPOCHS} completed in {epoch_time:.2f}s | Average Loss: {running_loss/len(train_loader):.4f}")

    print("Evaluating model...")
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"MLP Test Accuracy: {accuracy:.2f}%")

    # Export results
    result_file = './output/model_results.json'
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    
    results = {}
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                pass

    results['MLP_GPU'] = {'accuracy': float(f"{accuracy:.2f}"), 'train_time': float(f"{total_train_time:.2f}")}

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {result_file}")

if __name__ == "__main__":
    main()
