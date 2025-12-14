"""
多标签分类模型
支持EfficientNet-B4和DenseNet121作为backbone
"""

import torch
import torch.nn as nn
import timm


class MultiLabelClassifier(nn.Module):
    """
    多标签分类模型
    """
    
    def __init__(self, model_name="EfficientNet-B4", num_classes=14, pretrained=True):
        """
        Args:
            model_name: 模型名称 ("EfficientNet-B4" 或 "DenseNet121")
            num_classes: 类别数量
            pretrained: 是否使用预训练权重
        """
        super(MultiLabelClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # 加载backbone
        if model_name == "EfficientNet-B4":
            self.backbone = timm.create_model(
                'efficientnet_b4',
                pretrained=pretrained,
                num_classes=0,  # 移除分类头
                global_pool=''
            )
            # EfficientNet-B4的feature维度
            backbone_dim = self.backbone.num_features
        elif model_name == "DenseNet121":
            self.backbone = timm.create_model(
                'densenet121',
                pretrained=pretrained,
                num_classes=0,  # 移除分类头
                global_pool=''
            )
            # DenseNet121的feature维度
            backbone_dim = self.backbone.num_features
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(backbone_dim, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: 输入图像 (B, 3, H, W)
        Returns:
            logits: 分类logits (B, num_classes)
        """
        # 提取特征
        features = self.backbone.forward_features(x)
        
        # Global Average Pooling
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)
        
        # 分类
        logits = self.classifier(features)
        
        return logits
    
    def predict_proba(self, x):
        """
        预测概率（应用Sigmoid）
        Args:
            x: 输入图像
        Returns:
            probs: 概率 (B, num_classes)
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return probs


def create_model(model_name="EfficientNet-B4", num_classes=14, pretrained=True):
    """
    创建模型的便捷函数
    """
    model = MultiLabelClassifier(model_name, num_classes, pretrained)
    return model


if __name__ == "__main__":
    # 测试模型
    model = create_model("EfficientNet-B4", num_classes=14, pretrained=False)
    
    # 创建随机输入
    x = torch.randn(2, 3, 512, 512)
    
    # 前向传播
    logits = model(x)
    probs = model.predict_proba(x)
    
    print(f"输入形状: {x.shape}")
    print(f"Logits形状: {logits.shape}")
    print(f"概率形状: {probs.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")


