import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNNBranch(nn.Module):
    def __init__(self, in_channels=1):
        """
        增强的CNN分支，用于处理一种MRI图像（anat、dti或rest）。
        
        Args:
            in_channels (int): 输入图像的通道数，默认为1（灰度GIF图像）
        """
        super(CNNBranch, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 自适应池化层，确保输出固定大小
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
    def forward(self, x):
        """
        前向传播。
        
        Args:
            x (torch.Tensor): 输入图像，形状为 (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: 展平后的特征向量
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        
        # 自适应池化到固定大小 (4, 4)
        x = self.adaptive_pool(x)
        
        # 展平特征
        x = x.view(x.size(0), -1)  # (batch_size, 512 * 4 * 4 = 8192)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        """
        增强的Transformer Encoder，用于处理连接后的CNN特征。
        
        Args:
            d_model (int): 输入特征维度
            nhead (int): 注意力头数
            num_layers (int): Transformer层数
            dim_feedforward (int): 前馈网络隐藏层维度
            dropout (float): Dropout概率
        """
        super(TransformerEncoder, self).__init__()
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
    def forward(self, src):
        """
        前向传播。
        
        Args:
            src (torch.Tensor): 输入序列，形状为 (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Transformer输出，形状为 (batch_size, seq_len, d_model)
        """
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        位置编码模块。
        
        Args:
            d_model (int): 输入特征维度
            dropout (float): Dropout概率
            max_len (int): 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        添加位置编码。
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: 添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MRIModel(nn.Module):
    def __init__(self, in_channels=1, d_model=512, nhead=8, num_transformer_layers=4):
        """
        增强的MRI分类模型，包含三个CNN分支、Transformer Encoder和分类头。
        
        Args:
            in_channels (int): 输入图像通道数，默认为1
            d_model (int): Transformer输入特征维度
            nhead (int): Transformer注意力头数
            num_transformer_layers (int): Transformer层数
        """
        super(MRIModel, self).__init__()
        
        # 三个CNN分支
        self.anat_cnn = CNNBranch(in_channels)
        self.dti_cnn = CNNBranch(in_channels)
        self.rest_cnn = CNNBranch(in_channels)
        
        # 每个CNN分支的输出维度 (512 * 4 * 4 = 8192)
        cnn_output_dim = 512 * 4 * 4
        
        # 线性层将CNN输出映射到d_model
        self.fc_anat = nn.Linear(cnn_output_dim, d_model)
        self.fc_dti = nn.Linear(cnn_output_dim, d_model)
        self.fc_rest = nn.Linear(cnn_output_dim, d_model)
        
        # Transformer Encoder
        self.transformer = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_transformer_layers,
            dim_feedforward=2048
        )
        
        # 增强的分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # 二分类：患病(1) 或 正常(2)
        )
        
    def forward(self, anat, dti, rest):
        """
        前向传播。
        
        Args:
            anat (torch.Tensor): anat_1图像，形状为 (batch_size, in_channels, 256, 256)
            dti (torch.Tensor): dti_1图像，形状为 (batch_size, in_channels, 192, 192)
            rest (torch.Tensor): rest_1图像，形状为 (batch_size, in_channels, 64, 64)
            
        Returns:
            torch.Tensor: 分类输出，形状为 (batch_size, 2)
        """
        # 每个CNN分支处理对应的图像
        anat_features = self.anat_cnn(anat)
        dti_features = self.dti_cnn(dti)
        rest_features = self.rest_cnn(rest)
        
        # 映射到d_model维度
        anat_features = self.fc_anat(anat_features)
        dti_features = self.fc_dti(dti_features)
        rest_features = self.fc_rest(rest_features)
        
        # 将三个特征组成序列，形状为 (batch_size, seq_len=3, d_model)
        features = torch.stack([anat_features, dti_features, rest_features], dim=1)
        
        # Transformer处理
        transformer_output = self.transformer(features)
        
        # 取序列的平均值作为分类特征
        pooled_output = transformer_output.mean(dim=1)  # (batch_size, d_model)
        
        # 分类头
        output = self.classifier(pooled_output)
        
        return output

# 示例用法：
if __name__ == "__main__":
    # 假设输入图像大小分别为 256x256, 192x192, 64x64，单通道
    model = MRIModel(in_channels=1, d_model=512, nhead=16, num_transformer_layers=8)
    
    # 模拟输入数据
    batch_size = 8
    anat = torch.randn(batch_size, 1, 256, 256)
    dti = torch.randn(batch_size, 1, 192, 192)
    rest = torch.randn(batch_size, 1, 64, 64)
    
    # 前向传播
    output = model(anat, dti, rest)
    print(f"Output shape: {output.shape}")  # 应为 (batch_size, 2)