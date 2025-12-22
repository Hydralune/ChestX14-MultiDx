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
    
    支持多种模型大小：
    - 'tiny': Swin-T (约28M参数)
    - 'small': Swin-S (约50M参数)
    - 'base': Swin-B (约88M参数)
    """
    
    def __init__(self, num_classes=14, pretrained=True, input_channel_mode='expand', 
                 img_size=224, model_size='tiny'):
        """
        Args:
            num_classes: 类别数量
            pretrained: 是否使用ImageNet预训练权重
            input_channel_mode: 输入通道处理方式
                - "expand": 将灰度图扩展到3通道（需要在transform中处理）
                - "modify": 修改第一层适配1通道输入
            img_size: 输入图像尺寸
            model_size: 模型大小 ('tiny', 'small', 'base')
        """
        super(SwinMultiLabelClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.input_channel_mode = input_channel_mode
        self.model_size = model_size
        
        # 加载 Swin Transformer backbone
        self.backbone = timm.create_model(
            self._get_model_name(),
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
            img_size=img_size
        )
        
        # 获取 backbone 输出维度
        backbone_dim = None
        try:
            backbone_dim = self.backbone.num_features
        except:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, img_size, img_size)
                dummy_features = self.backbone.forward_features(dummy_input)
                if isinstance(dummy_features, (list, tuple)):
                    features = dummy_features[-1]
                else:
                    features = dummy_features
                if len(features.shape) == 4:
                    spatial_dims = features.shape[1:-1]
                    last_dim = features.shape[-1]
                    if all(d <= last_dim for d in spatial_dims):
                        backbone_dim = last_dim      # [B, H, W, C]
                    else:
                        backbone_dim = features.shape[1]  # [B, C, H, W]
                elif len(features.shape) == 3:
                    backbone_dim = features.shape[-1]
                else:
                    backbone_dim = features.shape[-1]
        
        if backbone_dim is None:
            # 根据模型大小设置默认特征维度
            if model_size == 'tiny':
                backbone_dim = 768
            elif model_size == 'small':
                backbone_dim = 768
            elif model_size == 'base':
                backbone_dim = 1024
            else:
                backbone_dim = 768
        
        self.backbone_dim = backbone_dim
    
    def _get_model_name(self):
        """根据model_size获取timm模型名称"""
        model_map = {
            'tiny': 'swin_tiny_patch4_window7_224',
            'small': 'swin_small_patch4_window7_224',
            'base': 'swin_base_patch4_window7_224'
        }
        return model_map.get(self.model_size, 'swin_tiny_patch4_window7_224')
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(backbone_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        if input_channel_mode == 'modify':
            self._modify_first_layer()
    
    def _modify_first_layer(self):
        """修改第一层以适配1通道输入"""
        if hasattr(self.backbone, 'patch_embed'):
            old_conv = self.backbone.patch_embed.proj
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding
            )
            with torch.no_grad():
                new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
                if old_conv.bias is not None:
                    new_conv.bias.data = old_conv.bias.data
            self.backbone.patch_embed.proj = new_conv
        else:
            raise ValueError("无法找到backbone的第一层，请检查模型结构")
    
    def forward(self, x):
        """
        x:
          - expand: (B, 3, H, W)
          - modify: (B, 1, H, W)
        """
        if self.input_channel_mode == 'modify' and x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        
        features = self.backbone.forward_features(x)
        
        if isinstance(features, (list, tuple)):
            features = features[-1]
        
        if len(features.shape) == 4:
            if features.shape[1] < features.shape[-1]:
                features = features.permute(0, 3, 1, 2)
            features = self.global_pool(features)
            features = features.view(features.size(0), -1)
        elif len(features.shape) == 3:
            features = features.mean(dim=1)
        elif len(features.shape) == 2:
            pass
        else:
            features = features.view(features.size(0), -1)
        
        logits = self.classifier(features)
        return logits
    
    def predict_proba(self, x):
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return probs

def create_model(num_classes=14, pretrained=True, input_channel_mode='expand', 
                 img_size=224, model_size='tiny'):
    """
    创建模型的便捷函数
    
    Args:
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
        input_channel_mode: 输入通道处理方式
        img_size: 输入图像尺寸
        model_size: 模型大小 ('tiny', 'small', 'base')
    """
    model = SwinMultiLabelClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        input_channel_mode=input_channel_mode,
        img_size=img_size,
        model_size=model_size
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

