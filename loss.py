"""
损失函数模块
支持Weighted BCE和Focal Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """
    加权二分类交叉熵损失
    用于处理类别不平衡问题
    """
    
    def __init__(self, pos_weight=None, reduction='mean'):
        """
        Args:
            pos_weight: 正样本权重 (num_classes,)
            reduction: 缩减方式 ('mean' 或 'sum')
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        
    def forward(self, logits, targets):
        """
        Args:
            logits: 模型输出 (B, num_classes)
            targets: 真实标签 (B, num_classes)
        Returns:
            loss: 损失值
        """
        # 计算BCE with logits（数值稳定）
        loss = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction='none'
        )
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Focal Loss
    用于处理类别不平衡和难样本挖掘
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: 平衡因子
            gamma: 聚焦参数
            reduction: 缩减方式
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits, targets):
        """
        Args:
            logits: 模型输出 (B, num_classes)
            targets: 真实标签 (B, num_classes)
        Returns:
            loss: 损失值
        """
        # 计算概率
        probs = torch.sigmoid(logits)
        
        # 计算BCE
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 计算p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # 计算focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # 应用alpha权重
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal Loss
        focal_loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def create_loss(loss_type='weighted_bce', pos_weight=None, 
                focal_alpha=0.25, focal_gamma=2.0):
    """
    创建损失函数的便捷函数
    
    Args:
        loss_type: 损失类型 ('weighted_bce' 或 'focal_loss')
        pos_weight: 正样本权重（用于weighted_bce）
        focal_alpha: Focal Loss的alpha参数
        focal_gamma: Focal Loss的gamma参数
    """
    if loss_type == 'weighted_bce':
        return WeightedBCELoss(pos_weight=pos_weight)
    elif loss_type == 'focal_loss':
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    else:
        raise ValueError(f"不支持的损失类型: {loss_type}")


if __name__ == "__main__":
    # 测试损失函数
    batch_size = 4
    num_classes = 14
    
    # 创建随机logits和targets
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    # 测试Weighted BCE
    pos_weight = torch.ones(num_classes) * 2.0
    weighted_bce = WeightedBCELoss(pos_weight=pos_weight)
    loss1 = weighted_bce(logits, targets)
    print(f"Weighted BCE Loss: {loss1.item():.4f}")
    
    # 测试Focal Loss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss2 = focal_loss(logits, targets)
    print(f"Focal Loss: {loss2.item():.4f}")


