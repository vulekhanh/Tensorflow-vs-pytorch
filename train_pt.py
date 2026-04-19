import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["accuracy", "time"])
    parser.add_argument("--target_acc", type=float, default=0.55)
    parser.add_argument("--time_limit", type=int, default=600)
    args = parser.parse_args()

    MODE = args.mode
    TARGET_ACC = args.target_acc
    TIME_LIMIT = args.time_limit

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()

    results = []

    for epoch in range(50):
        model.train()
        running_loss = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # evaluation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1} | Loss: {running_loss:.2f} | Acc: {acc:.4f} | Time: {elapsed:.2f}s")

        results.append([epoch+1, acc, elapsed])

        if MODE == "accuracy" and acc >= TARGET_ACC:
            print("Reached target accuracy")
            break

        if MODE == "time" and elapsed >= TIME_LIMIT:
            print("Reached time limit")
            break

    df = pd.DataFrame(results, columns=["Epoch", "Accuracy", "Time"])
    df.to_csv(f"results_pt_{MODE}.csv", index=False)

    print("Saved results!")

if __name__ == "__main__":
    main()
