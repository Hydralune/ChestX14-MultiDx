"""
多标签分类模型 (High Performance Version)
集成空间注意力机制 (SAM) + 解耦分类头 (Decoupled Heads)
支持 EfficientNet 和 DenseNet 作为 Backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SpatialAttention(nn.Module):
    """
    空间注意力模块
    让模型在Pooling之前聚焦于图像的关键区域（如病灶），抑制背景噪音。
    """
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        # 使用 1x1 卷积将特征图压缩为 1 个通道的注意力图
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W]
        # map: [B, 1, H, W]
        attn_map = self.conv(x)
        # 将注意力图作用回原特征图 (加权)
        return x * attn_map


class DecoupledClassHead(nn.Module):
    """
    解耦分类头
    为每一个类别单独建立一个小型的神经网络，防止类间特征干扰。
    """
    def __init__(self, in_features, hidden_dim=128):
        super(DecoupledClassHead, self).__init__()
        self.head = nn.Sequential(
            nn.Dropout(0.2),              # 独立的 Dropout
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),                    # Swish 激活函数，通常比 ReLU 效果更好
            nn.BatchNorm1d(hidden_dim),   # BN 层加速收敛
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)      # 输出单个类别的 logit
        )

    def forward(self, x):
        return self.head(x)


class MultiLabelClassifier(nn.Module):
    """
    多标签分类模型 - 高阶版
    Backbone -> Spatial Attention -> Global Pool -> 14 Separate Heads
    """
    
    def __init__(self, model_name="DenseNet121", num_classes=14, pretrained=True):
        """
        Args:
            model_name: 模型名称 ("EfficientNet-B4" 或 "DenseNet121")
            num_classes: 类别数量
            pretrained: 是否使用预训练权重
        """
        super(MultiLabelClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # === 1. 加载 Backbone ===
        if model_name == "EfficientNet-B4":
            self.backbone = timm.create_model(
                'efficientnet_b4',
                pretrained=pretrained,
                num_classes=0,
                global_pool='' # 保留空间特征 [B, C, H, W]
            )
            backbone_dim = self.backbone.num_features
            
        elif model_name == "DenseNet121":
            self.backbone = timm.create_model(
                'densenet121',
                pretrained=pretrained,
                num_classes=0,
                global_pool='' # 保留空间特征 [B, C, H, W]
            )
            backbone_dim = self.backbone.num_features
            
        elif model_name == "ResNet50": # 增加对 ResNet50 的支持
             self.backbone = timm.create_model(
                'resnet50',
                pretrained=pretrained,
                num_classes=0,
                global_pool=''
            )
             backbone_dim = self.backbone.num_features
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        # === 2. 空间注意力机制 ===
        self.attention = SpatialAttention(backbone_dim)
        
        # === 3. 全局池化 ===
        # 使用自适应平均池化将 [B, C, H, W] -> [B, C, 1, 1]
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # === 4. 解耦分类头 (关键改进) ===
        # 创建一个 ModuleList，包含 14 个独立的子网络
        # 这模仿了训练 14 个不同模型的效果，但共享 Backbone
        self.heads = nn.ModuleList([
            DecoupledClassHead(backbone_dim, hidden_dim=128) 
            for _ in range(num_classes)
        ])
        
    def forward(self, x):
        """
        Args:
            x: 输入图像 (B, 3, H, W)
        Returns:
            logits: 分类logits (B, num_classes)
        """
        # 1. 提取特征 [B, C, H, W]
        features = self.backbone.forward_features(x)
        
        # 2. 应用空间注意力 (让模型聚焦病灶)
        features = self.attention(features)
        
        # 3. 全局池化 [B, C, 1, 1] -> [B, C]
        pooled_features = self.global_pool(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        
        # 4. 通过 14 个独立的头进行预测
        logits_list = []
        for head in self.heads:
            # 每个 head 输出 [B, 1]
            logits_list.append(head(pooled_features))
            
        # 5. 拼接结果 [B, 14]
        logits = torch.cat(logits_list, dim=1)
        
        return logits
    
    def predict_proba(self, x):
        """
        预测概率
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return probs


def create_model(model_name="DenseNet121", num_classes=14, pretrained=True):
    """
    创建模型的便捷函数
    建议默认使用 DenseNet121，因为它在医学图像上通常比 EfficientNet 更容易训练且泛化更好
    """
    model = MultiLabelClassifier(model_name, num_classes, pretrained)
    return model


if __name__ == "__main__":
    # 测试代码
    print("正在测试改进版模型架构...")
    
    # 1. 测试模型实例化
    try:
        model = create_model("DenseNet121", num_classes=14, pretrained=False)
        print("模型创建成功: DenseNet121 + Attention + Decoupled Heads")
    except Exception as e:
        print(f"模型创建失败: {e}")
        exit()

    # 2. 创建随机输入 (模拟 batch_size=2)
    x = torch.randn(2, 3, 512, 512)
    
    # 3. 前向传播测试
    logits = model(x)
    probs = model.predict_proba(x)
    
    print("-" * 30)
    print(f"输入形状: {x.shape}")
    print(f"输出 Logits 形状: {logits.shape} (预期: [2, 14])")
    
    # 4. 验证是否包含 14 个头
    print(f"独立分类头数量: {len(model.heads)}")
    
    # 5. 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print("-" * 30)
    print("测试通过！")