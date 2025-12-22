"""
Swin Transformer 多标签分类模型
使用 timm 的 swin_tiny_patch4_window7_224 作为 backbone
"""

import torch
import torch.nn as nn
import timm


class SwinMultiLabelClassifier(nn.Module):
    """
    基于 Swin Transformer 的多标签分类模型
    
    支持两种输入通道处理方式：
    1. expand: 将1通道灰度图扩展到3通道（复制3次）
    2. modify: 修改第一层适配1通道输入
    """
    
    def __init__(self, num_classes=14, pretrained=True, input_channel_mode='expand'):
        """
        Args:
            num_classes: 类别数量
            pretrained: 是否使用ImageNet预训练权重
            input_channel_mode: 输入通道处理方式
                - "expand": 将灰度图扩展到3通道（需要在transform中处理）
                - "modify": 修改第一层适配1通道输入
        """
        super(SwinMultiLabelClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.input_channel_mode = input_channel_mode
        
        # 加载 Swin-T backbone
        # 注意：不设置global_pool，让forward_features返回特征图
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            num_classes=0,  # 不使用分类头
            global_pool=''  # 不进行全局池化，保留空间特征
        )
        
        # 获取backbone的特征维度
        # Swin-T的特征维度是768
        backbone_dim = None
        try:
            backbone_dim = self.backbone.num_features
        except:
            # 如果无法获取，使用dummy输入测试
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                dummy_features = self.backbone.forward_features(dummy_input)
                if isinstance(dummy_features, (list, tuple)):
                    # 如果返回多个stage的特征，取最后一个
                    features = dummy_features[-1]
                else:
                    features = dummy_features
                
                # 如果是4D tensor [B, C, H, W]，取C维度
                if len(features.shape) == 4:
                    backbone_dim = features.shape[1]
                elif len(features.shape) == 3:
                    # [B, N, C] 格式，取C维度
                    backbone_dim = features.shape[-1]
                else:
                    backbone_dim = features.shape[-1]
        
        # 如果还是无法获取，使用默认值（Swin-T的特征维度是768）
        if backbone_dim is None:
            backbone_dim = 768  # Swin-T的默认特征维度
        
        self.backbone_dim = backbone_dim
        
        # 全局平均池化（如果forward_features返回的是4D特征图）
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(backbone_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # 如果使用modify模式，需要修改第一层
        if input_channel_mode == 'modify':
            self._modify_first_layer()
    
    def _modify_first_layer(self):
        """修改第一层以适配1通道输入"""
        # 获取第一层（patch embedding层）
        # Swin的第一层通常是PatchEmbed
        if hasattr(self.backbone, 'patch_embed'):
            old_conv = self.backbone.patch_embed.proj
            # 创建新的1通道卷积层
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding
            )
            # 初始化：将3通道权重的平均值复制到1通道
            with torch.no_grad():
                new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
                if old_conv.bias is not None:
                    new_conv.bias.data = old_conv.bias.data
            self.backbone.patch_embed.proj = new_conv
        else:
            raise ValueError("无法找到backbone的第一层，请检查模型结构")
    
    def forward(self, x):
        """
        Args:
            x: 输入图像
                - 如果input_channel_mode='expand': (B, 3, H, W)
                - 如果input_channel_mode='modify': (B, 1, H, W)
        Returns:
            logits: 分类logits (B, num_classes)
        """
        # 如果使用modify模式但输入是3通道，需要转换
        if self.input_channel_mode == 'modify' and x.shape[1] == 3:
            # 转换为灰度图（取RGB的平均值）
            x = x.mean(dim=1, keepdim=True)
        
        # 提取特征
        # Swin的forward_features可能返回：
        # 1. 4D tensor [B, C, H, W] - 特征图
        # 2. 3D tensor [B, N, C] - token序列（Swin的输出格式）
        # 3. 2D tensor [B, C] - 已经pooled的特征
        features = self.backbone.forward_features(x)
        
        # 处理不同格式的特征
        if isinstance(features, (list, tuple)):
            # 如果返回多个stage的特征，取最后一个
            features = features[-1]
        
        # 处理特征格式
        if len(features.shape) == 4:
            # [B, C, H, W] -> [B, C] 使用全局平均池化
            features = self.global_pool(features)
            features = features.view(features.size(0), -1)
        elif len(features.shape) == 3:
            # [B, N, C] -> [B, C] 对token维度求平均
            features = features.mean(dim=1)
        elif len(features.shape) == 2:
            # 已经是 [B, C]
            pass
        else:
            # 其他情况，尝试flatten
            features = features.view(features.size(0), -1)
        
        # 分类
        logits = self.classifier(features)
        
        return logits
    
    def predict_proba(self, x):
        """预测概率"""
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return probs


def create_model(num_classes=14, pretrained=True, input_channel_mode='expand'):
    """
    创建模型的便捷函数
    
    Args:
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
        input_channel_mode: 输入通道处理方式
    """
    model = SwinMultiLabelClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        input_channel_mode=input_channel_mode
    )
    return model


if __name__ == "__main__":
    # 测试模型
    print("测试 Swin-T 模型...")
    
    # 测试expand模式（3通道输入）
    print("\n1. 测试 expand 模式（3通道输入）...")
    model_expand = create_model(num_classes=14, pretrained=False, input_channel_mode='expand')
    x_expand = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        logits_expand = model_expand(x_expand)
        probs_expand = model_expand.predict_proba(x_expand)
    print(f"   输入形状: {x_expand.shape}")
    print(f"   输出logits形状: {logits_expand.shape}")
    print(f"   输出概率形状: {probs_expand.shape}")
    
    # 测试modify模式（1通道输入）
    print("\n2. 测试 modify 模式（1通道输入）...")
    model_modify = create_model(num_classes=14, pretrained=False, input_channel_mode='modify')
    x_modify = torch.randn(2, 1, 512, 512)
    with torch.no_grad():
        logits_modify = model_modify(x_modify)
        probs_modify = model_modify.predict_proba(x_modify)
    print(f"   输入形状: {x_modify.shape}")
    print(f"   输出logits形状: {logits_modify.shape}")
    print(f"   输出概率形状: {probs_modify.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model_expand.parameters())
    trainable_params = sum(p.numel() for p in model_expand.parameters() if p.requires_grad)
    print(f"\n模型参数量 - 总数: {total_params:,}, 可训练: {trainable_params:,}")
    
    print("\n测试通过！")

