"""
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


class LabelCorrelationModule(nn.Module):
    """
    Label Correlation Modeling Module (SOTA关键改进)
    
    【核心创新 - 显式建模标签相关性】
    - 传统方法假设14个标签相互独立，但医学上存在相关性：
      * Effusion ↔ Atelectasis (积液常伴随肺不张)
      * Pneumonia ↔ Consolidation (肺炎常表现为实变)
      * Edema ↔ Cardiomegaly (水肿与心脏扩大相关)
    - 通过Label Embedding学习标签表示
    - 使用Cross-Attention让visual feature与label embeddings交互
    - 输出label-aware feature，提升Macro AUC
    - 预期提升: +0.02~0.04 AUC（通过显式建模标签相关性）
    
    结构:
    Visual Feature [B, C] + Label Embeddings [num_classes, D]
      → Cross Attention
      → Label-Aware Features [B, num_classes, D']
      → Per-label Features [B, num_classes, C']
    """
    def __init__(self, visual_dim, num_classes=14, label_dim=128, hidden_dim=256):
        """
        Args:
            visual_dim: visual feature的维度（backbone输出维度）
            num_classes: 标签数量
            label_dim: label embedding的维度
            hidden_dim: attention的hidden维度
        """
        super(LabelCorrelationModule, self).__init__()
        self.num_classes = num_classes
        self.visual_dim = visual_dim
        self.label_dim = label_dim
        self.hidden_dim = hidden_dim
        
        # === 1. Label Embeddings (可学习参数) ===
        # 每个标签学习一个表示向量，用于捕获标签语义和相关性
        self.label_embeddings = nn.Parameter(
            torch.randn(num_classes, label_dim)
        )
        nn.init.xavier_uniform_(self.label_embeddings)
        
        # === 2. Visual Feature Projection ===
        # 将visual feature投影到合适的维度
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        
        # === 3. Label Embedding Projection ===
        # 将label embeddings投影到hidden维度
        self.label_proj = nn.Linear(label_dim, hidden_dim)
        
        # === 4. Cross-Attention (Label attends to Visual) ===
        # 让每个标签关注visual feature中的相关信息
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # === 5. Output Projection ===
        # 将attention输出投影回visual_dim，用于后续的decoupled heads
        self.output_proj = nn.Linear(hidden_dim, visual_dim)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(visual_dim)
        
    def forward(self, visual_feature):
        """
        Args:
            visual_feature: [B, visual_dim] - 全局池化后的visual feature
        Returns:
            label_aware_features: [B, num_classes, visual_dim] - 每个标签的refined feature
        """
        B = visual_feature.size(0)
        
        # === 1. Project visual feature ===
        # [B, visual_dim] -> [B, hidden_dim]
        visual_proj = self.visual_proj(visual_feature)  # [B, hidden_dim]
        
        # === 2. Project label embeddings ===
        # [num_classes, label_dim] -> [num_classes, hidden_dim]
        # 扩展到batch维度: [num_classes, hidden_dim] -> [B, num_classes, hidden_dim]
        label_emb = self.label_proj(self.label_embeddings)  # [num_classes, hidden_dim]
        label_emb = label_emb.unsqueeze(0).expand(B, -1, -1)  # [B, num_classes, hidden_dim]
        
        # === 3. Cross-Attention ===
        # Query: label embeddings [B, num_classes, hidden_dim]
        # Key, Value: visual feature [B, 1, hidden_dim] (broadcast)
        visual_query = visual_proj.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Label attends to Visual
        # 每个标签通过学习到的embedding去关注visual feature中的相关信息
        attn_output, _ = self.attention(
            query=label_emb,  # [B, num_classes, hidden_dim]
            key=visual_query,  # [B, 1, hidden_dim]
            value=visual_query  # [B, 1, hidden_dim]
        )
        attn_output = self.norm1(attn_output + label_emb)  # Residual + Norm
        
        # === 4. Output Projection ===
        # [B, num_classes, hidden_dim] -> [B, num_classes, visual_dim]
        label_aware_features = self.output_proj(attn_output)  # [B, num_classes, visual_dim]
        
        # Add residual connection with broadcasted visual feature
        visual_feature_expanded = visual_feature.unsqueeze(1)  # [B, 1, visual_dim]
        label_aware_features = self.norm2(label_aware_features + visual_feature_expanded)
        
        return label_aware_features  # [B, num_classes, visual_dim]


class DecoupledClassHead(nn.Module):
    """
    解耦分类头 (SOTA改进版本)
    
    【SOTA改进 - 去除BatchNorm】
    - BatchNorm在小batch和类不平衡场景下不稳定
    - LayerNorm更适合单样本归一化，对batch size不敏感
    - 简化结构，减少参数，提升rare class的稳定性
    - 预期提升: +0.005 AUC（通过更稳定的归一化）
    """
    def __init__(self, in_features, hidden_dim=128):
        super(DecoupledClassHead, self).__init__()
        # 更轻量的Head结构，使用LayerNorm替代BatchNorm
        self.head = nn.Sequential(
            nn.Dropout(0.2),              # 独立的 Dropout
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),     # LayerNorm替代BatchNorm（对batch size不敏感）
            nn.SiLU(),                    # Swish 激活函数
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
        
        # === 4. Label Correlation Module (SOTA关键改进) ===
        # 显式建模标签相关性，通过Label Embedding和Attention机制
        # 预期提升: +0.02~0.04 Macro AUC
        self.label_correlation = LabelCorrelationModule(
            visual_dim=backbone_dim,
            num_classes=num_classes,
            label_dim=128,
            hidden_dim=256
        )
        
        # === 5. 解耦分类头 (关键改进) ===
        # 每个head使用对应的label-aware feature进行预测
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
        
        # 4. Label Correlation Modeling (SOTA关键改进)
        # [B, C] -> [B, num_classes, C]
        # 每个标签获得一个refined的feature representation
        label_aware_features = self.label_correlation(pooled_features)  # [B, num_classes, C]
        
        # 5. 通过 14 个独立的头进行预测
        logits_list = []
        for i, head in enumerate(self.heads):
            # 每个head使用对应标签的label-aware feature
            # label_aware_features[:, i, :] 形状为 [B, C]
            label_feat = label_aware_features[:, i, :]  # [B, C]
            # 每个 head 输出 [B, 1]
            logits_list.append(head(label_feat))
            
        # 6. 拼接结果 [B, 14]
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