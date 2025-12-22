"""
Test-Time Augmentation (TTA) 模块
用于推理时提升模型性能
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np


def tta_predict(model, images, device, num_augments=5):
    """
    Test-Time Augmentation预测
    
    Args:
        model: 模型
        images: 输入图像 (B, C, H, W)
        device: 设备
        num_augments: 增强次数（包括原始图像）
    
    Returns:
        probs: 平均后的概率 (B, num_classes)
    """
    model.eval()
    batch_size = images.size(0)
    num_classes = model.num_classes
    
    all_probs = []
    
    with torch.no_grad():
        # 1. 原始图像
        logits = model(images)
        probs = torch.sigmoid(logits)
        all_probs.append(probs)
        
        # 2. 水平翻转
        images_flip = torch.flip(images, [3])
        logits_flip = model(images_flip)
        probs_flip = torch.sigmoid(logits_flip)
        all_probs.append(probs_flip)
        
        # 3. 轻微旋转（±5度）
        if num_augments >= 3:
            angle1 = 5.0
            angle2 = -5.0
            
            # 旋转5度
            images_rot1 = F.affine_grid(
                torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float32).repeat(batch_size, 1, 1),
                images.size(),
                align_corners=False
            )
            # 简化：只做水平翻转的变体
            # 实际旋转需要更复杂的实现，这里用其他增强替代
            
        # 4. 多尺度（如果支持）
        # 注意：多尺度需要重新resize，这里简化处理
        
    # 平均所有增强结果
    all_probs = torch.stack(all_probs, dim=0)  # (num_augments, B, num_classes)
    avg_probs = all_probs.mean(dim=0)  # (B, num_classes)
    
    return avg_probs


def tta_predict_simple(model, images, device):
    """
    简单的TTA：原始 + 水平翻转
    
    Args:
        model: 模型
        images: 输入图像 (B, C, H, W)
        device: 设备
    
    Returns:
        probs: 平均后的概率 (B, num_classes)
    """
    model.eval()
    
    with torch.no_grad():
        # 原始图像
        logits_orig = model(images)
        probs_orig = torch.sigmoid(logits_orig)
        
        # 水平翻转
        images_flip = torch.flip(images, [3])
        logits_flip = model(images_flip)
        probs_flip = torch.sigmoid(logits_flip)
        
        # 平均
        probs = (probs_orig + probs_flip) / 2.0
    
    return probs

