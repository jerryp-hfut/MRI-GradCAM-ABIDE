import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from data_loader import MRIDataset
from model import MRICNN

def plot_roc_curve(y_true, y_scores, save_path="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    # Set figure size to 5x5 inches before plotting
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curve(y_true, y_scores, save_path="precision_recall_curve.png"):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    # Set figure size to 5x5 inches before plotting
    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    
    # Set figure size to 5x5 inches before plotting
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data parameters
    data_dir = "datas_mri"
    csv_file = "labels.csv"
    
    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load dataset
    dataset = MRIDataset(data_dir, csv_file, transform=transform)
    print(f"Total valid samples: {len(dataset)}")
    
    # Split dataset (8:1:1)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Create test data loader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load model
    model = MRICNN().to(device)
    model.load_state_dict(torch.load("best_model_anat.pth", map_location=device))
    model.eval()
    
    # Collect predictions and true labels
    y_true = []
    y_scores = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)[:, 1]  # Probability for class 1 (diseased)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_scores.extend(probabilities.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = np.array(y_pred)
    
    # Create output directory
    output_dir = "performance_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save plots
    plot_roc_curve(y_true, y_scores, save_path=os.path.join(output_dir, "roc_curve.png"))
    plot_precision_recall_curve(y_true, y_scores, save_path=os.path.join(output_dir, "precision_recall_curve.png"))
    plot_confusion_matrix(y_true, y_pred, save_path=os.path.join(output_dir, "confusion_matrix.png"))
    
    print(f"Visualizations saved in {output_dir}")

if __name__ == "__main__":
    main()
