import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from data_loader import MRIDataLoader
from model import MRIModel
import random

class MRIDataset(Dataset):
    def __init__(self, data, transform_anat=None, transform_dti=None, transform_rest=None):
        """
        自定义数据集，用于加载MRI图像和标签。
        
        Args:
            data (list): 从 data_loader 获得的样本列表，格式为 [(sample_id, image_paths, label), ...]
            transform_anat: anat 图像变换
            transform_dti: dti 图像变换
            transform_rest: rest 图像变换
        """
        self.data = data
        self.transform_anat = transform_anat
        self.transform_dti = transform_dti
        self.transform_rest = transform_rest
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample_id, image_paths, label = self.data[idx]
        
        # 加载图像
        anat_img = Image.open(image_paths['anat'])# .convert('L')  # 转换为灰度图像
        dti_img = Image.open(image_paths['dti'])# .convert('L')
        rest_img = Image.open(image_paths['rest'])# .convert('L')
        
        # 应用变换
        if self.transform_anat:
            anat_img = self.transform_anat(anat_img)
        if self.transform_dti:
            dti_img = self.transform_dti(dti_img)
        if self.transform_rest:
            rest_img = self.transform_rest(rest_img)
        
        # 将标签转换为 0 (患病) 和 1 (正常) 以匹配 CrossEntropyLoss
        label = 0 if label == 1 else 1
        
        return anat_img, dti_img, rest_img, label

def train_model():
    # 数据加载
    data_loader = MRIDataLoader(data_dir="datas_mri", labels_csv="lables.csv")
    dataset, stats = data_loader.load_data()
    
    print(f"可用样本总数量: {stats['total_samples']}")
    print(f"正常样本数量 (DX_GROUP=2): {stats['normal_samples']}")
    print(f"患病样本数量 (DX_GROUP=1): {stats['disease_samples']}")
    
    # 数据集划分：训练集 8/11，验证集 1/11，测试集 2/11
    total_samples = len(dataset)
    train_size = int(7/10 * total_samples)
    val_size = int(1/10 * total_samples)
    test_size = total_samples - train_size - val_size
    
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
    
    # 定义不同图像的变换
    transform_anat = transforms.Compose([
        transforms.Resize((256, 256)),  # anat 图像调整为 256x256
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    transform_dti = transforms.Compose([
        transforms.Resize((192, 192)),  # dti 图像调整为 192x192
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    transform_rest = transforms.Compose([
        transforms.Resize((64, 64)),  # rest 图像调整为 64x64
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 创建数据集和数据加载器
    train_dataset = MRIDataset(train_data, transform_anat, transform_dti, transform_rest)
    val_dataset = MRIDataset(val_data, transform_anat, transform_dti, transform_rest)
    test_dataset = MRIDataset(test_data, transform_anat, transform_dti, transform_rest)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # 初始化模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MRIModel(in_channels=1, d_model=8, nhead=4, num_transformer_layers=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
    # 训练参数
    num_epochs = 100
    best_val_acc = 0.0
    best_model_path = "best_model.pth"
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 训练进度条
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for anat, dti, rest, labels in train_bar:
            anat, dti, rest, labels = anat.to(device), dti.to(device), rest.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(anat, dti, rest)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * anat.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix({"Loss": loss.item(), "Acc": 100. * train_correct / train_total})
        
        train_loss /= train_total
        train_acc = 100. * train_correct / train_total
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for anat, dti, rest, labels in val_bar:
                anat, dti, rest, labels = anat.to(device), dti.to(device), rest.to(device), labels.to(device)
                
                outputs = model(anat, dti, rest)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * anat.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_bar.set_postfix({"Loss": loss.item(), "Acc": 100. * val_correct / val_total})
        
        val_loss /= val_total
        val_acc = 100. * val_correct / val_total
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"保存最佳模型: Epoch {epoch+1}, Val Acc: {val_acc:.2f}%")
        
        # 每5个epoch在测试集上评估
        if (epoch + 1) % 5 == 0:
            model.eval()
            test_correct = 0
            test_total = 0
            
            test_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]")
            with torch.no_grad():
                for anat, dti, rest, labels in test_bar:
                    anat, dti, rest, labels = anat.to(device), dti.to(device), rest.to(device), labels.to(device)
                    
                    outputs = model(anat, dti, rest)
                    _, predicted = torch.max(outputs, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
                    
                    test_bar.set_postfix({"Acc": 100. * test_correct / test_total})
            
            test_acc = 100. * test_correct / test_total
            print(f"测试集准确率: {test_acc:.2f}%")
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    train_model()