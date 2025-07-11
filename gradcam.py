import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
from data_loader import MRIDataLoader
from model import MRIModel

class GradCAM:
    def __init__(self, model, target_layers):
        """
        初始化 Grad-CAM。
        
        Args:
            model: 训练好的模型（MRIModel）
            target_layers: 目标层列表（例如 [model.anat_cnn.conv5, model.dti_cnn.conv5, model.rest_cnn.conv5]）
        """
        self.model = model
        self.target_layers = target_layers
        self.gradients = []
        self.activations = []
        
        # 注册前向和反向钩子
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations.append(output.detach())
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients.append(grad_out[0].detach())
        
        for layer in self.target_layers:
            layer.register_forward_hook(forward_hook)
            layer.register_backward_hook(backward_hook)
    
    def generate_heatmap(self, input_tensor, class_idx=None, branch_idx=0):
        """
        为单个输入生成 Grad-CAM 热力图。
        
        Args:
            input_tensor (torch.Tensor): 输入图像，形状为 (1, in_channels, height, width)
            class_idx (int, optional): 目标类别索引，若为 None 则使用预测类别
            branch_idx (int): 分支索引（0: anat, 1: dti, 2: rest）
            
        Returns:
            numpy.ndarray: 热力图
        """
        self.model.eval()
        self.gradients = []
        self.activations = []
        
        # 前向传播
        input_tensor = input_tensor.to(next(self.model.parameters()).device)
        inputs = [
            input_tensor if branch_idx == 0 else torch.zeros_like(input_tensor),
            input_tensor if branch_idx == 1 else torch.zeros_like(input_tensor),
            input_tensor if branch_idx == 2 else torch.zeros_like(input_tensor)
        ]
        output = self.model(*inputs)
        
        # 获取目标类别
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        # 反向传播
        self.model.zero_grad()
        output[:, class_idx].backward()
        
        # 获取对应分支的激活和梯度
        activation = self.activations[branch_idx]
        gradient = self.gradients[branch_idx]
        
        # 计算全局平均池化权重
        weights = torch.mean(gradient, dim=[2, 3], keepdim=True)
        # 计算热力图
        heatmap = torch.sum(weights * activation, dim=1).squeeze().cpu().numpy()
        # ReLU 激活
        heatmap = np.maximum(heatmap, 0)
        # 归一化到 [0, 1]
        heatmap = heatmap / (heatmap.max() + 1e-8)
        
        return heatmap

def resize_heatmap(heatmap, size):
    """
    将热力图调整到目标尺寸。
    
    Args:
        heatmap (numpy.ndarray): 热力图
        size (tuple): 目标尺寸 (width, height)
        
    Returns:
        numpy.ndarray: 调整后的热力图
    """
    heatmap = cv2.resize(heatmap, size, interpolation=cv2.INTER_LINEAR)
    heatmap = np.clip(heatmap, 0, 1)
    return heatmap

def save_heatmap(heatmap, output_path, original_img_path):
    """
    将热力图叠加到原始图像上（7:3透明度）并保存为 PNG。
    
    Args:
        heatmap (numpy.ndarray): 热力图，值在 [0, 1]
        output_path (str): 保存路径
        original_img_path (str): 原始图像路径
    """
    # 加载原始图像并转换为 RGB
    original_img = Image.open(original_img_path).convert('RGB')
    original_img = np.array(original_img).astype(np.uint8)
    
    # 调整热力图到原始图像尺寸
    heatmap = resize_heatmap(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    # 转换为 0-255 的 uint8 并应用颜色映射
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 融合图像：热力图占 70%，原图占 30%
    alpha = 0.7  # 热力图透明度
    beta = 0.3   # 原图透明度
    blended = cv2.addWeighted(heatmap_colored, alpha, original_img, beta, 0.0)
    
    # 保存融合图像
    cv2.imwrite(output_path, blended)

def main():
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型并设置参数
    try:
        model = MRIModel(in_channels=1, d_model=8, nhead=4, num_transformer_layers=2).to(device)
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        model.eval()
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 加载数据
    try:
        data_loader = MRIDataLoader(data_dir="datas_mri", labels_csv="lables.csv")
        dataset, stats = data_loader.load_data()
        print(f"处理样本总数: {stats['total_samples']}")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    # 定义图像变换
    transform_anat = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    transform_dti = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    transform_rest = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 定义目标层
    try:
        target_layers = [model.anat_cnn.conv5, model.dti_cnn.conv5, model.rest_cnn.conv5]
    except AttributeError:
        print("模型结构不匹配：请确保 model.py 中 CNNBranch 包含 conv5 层")
        return
    
    # 初始化 Grad-CAM
    grad_cam = GradCAM(model, target_layers)
    
    # 处理每个样本
    for sample_id, image_paths, label in dataset:
        print(f"处理样本: {sample_id}")
        
        try:
            # 加载图像
            anat_img = Image.open(image_paths['anat']).convert('L')
            dti_img = Image.open(image_paths['dti']).convert('L')
            rest_img = Image.open(image_paths['rest']).convert('L')
            
            # 应用变换
            anat_tensor = transform_anat(anat_img).unsqueeze(0)  # (1, 1, 256, 256)
            dti_tensor = transform_dti(dti_img).unsqueeze(0)    # (1, 1, 192, 192)
            rest_tensor = transform_rest(rest_img).unsqueeze(0) # (1, 1, 64, 64)
            
            # 生成热力图
            anat_heatmap = grad_cam.generate_heatmap(anat_tensor, class_idx=None, branch_idx=0)
            dti_heatmap = grad_cam.generate_heatmap(dti_tensor, class_idx=None, branch_idx=1)
            rest_heatmap = grad_cam.generate_heatmap(rest_tensor, class_idx=None, branch_idx=2)
            
            # 保存热力图到与输入图像相同的路径
            anat_output_path = os.path.join(os.path.dirname(image_paths['anat']), "gradcam_anat.png")
            dti_output_path = os.path.join(os.path.dirname(image_paths['dti']), "gradcam_dti.png")
            rest_output_path = os.path.join(os.path.dirname(image_paths['rest']), "gradcam_rest.png")
            
            # 调整热力图尺寸并保存（叠加到原图）
            save_heatmap(anat_heatmap, anat_output_path, image_paths['anat'])
            save_heatmap(dti_heatmap, dti_output_path, image_paths['dti'])
            save_heatmap(rest_heatmap, rest_output_path, image_paths['rest'])
            
            print(f"热力图已保存: {anat_output_path}, {dti_output_path}, {rest_output_path}")
            
        except Exception as e:
            print(f"处理样本 {sample_id} 失败: {e}")
            continue

if __name__ == "__main__":
    main()