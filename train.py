import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from data_loader import MRIDataset
from model import MRICNN
import os

def train_model(model, train_loader, val_loader, test_loader, num_epochs=200, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_test_accuracy = 0.0  # Track the best test accuracy
    best_model_path = "best_model.pth"  # Path to save the best model
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for images, labels, _ in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.set_postfix(loss=running_loss/len(train_bar))
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        print(f'Epoch {epoch+1}, Validation Accuracy: {val_accuracy:.2f}%')
        
        # Test every 10 epochs
        if (epoch + 1) % 10 == 0:
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for images, labels, _ in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
            
            test_accuracy = 100 * test_correct / test_total
            print(f'Epoch {epoch+1}, Test Accuracy: {test_accuracy:.2f}%')
            
            # Save model if test accuracy is a new high
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                torch.save(model.state_dict(), best_model_path)
                print(f'New best test accuracy {test_accuracy:.2f}%. Model saved to {best_model_path}')

def main():
    # Data parameters
    data_dir = "datas_mri"
    csv_file = "labels.csv"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 单通道标准化
    ])
    dataset = MRIDataset(data_dir, csv_file, transform=transform)
    
    # Split dataset (8:1:1)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MRICNN().to(device)
    
    train_model(model, train_loader, val_loader, test_loader, num_epochs=200, device=device)

if __name__ == "__main__":
    main()
