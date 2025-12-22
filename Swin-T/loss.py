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


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification
    对负样本使用不同的gamma，更适合多标签分类任务
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, reduction='mean'):
        """
        Args:
            gamma_neg: 负样本的gamma（通常较大，如4）
            gamma_pos: 正样本的gamma（通常较小，如1）
            clip: 概率裁剪阈值，防止过度关注难样本
            eps: 数值稳定性参数
            reduction: 缩减方式
        """
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: 模型输出 (B, num_classes)
            targets: 真实标签 (B, num_classes)
        """
        # 计算概率
        probs = torch.sigmoid(logits)
        
        # 裁剪概率，防止过度关注难样本
        if self.clip is not None and self.clip > 0:
            pt = torch.clamp(probs, self.clip, 1 - self.clip)
        else:
            pt = probs
        
        # 分别处理正负样本
        targets_pos = targets
        targets_neg = 1 - targets
        
        # 正样本的focal weight: (1 - pt)^gamma_pos
        pt_pos = pt * targets_pos + (1 - pt) * (1 - targets_pos)
        pt_pos = torch.clamp(pt_pos, min=self.eps, max=1-self.eps)
        focal_weight_pos = (1 - pt_pos) ** self.gamma_pos
        
        # 负样本的focal weight: pt^gamma_neg
        pt_neg = pt * targets_neg + (1 - pt) * (1 - targets_neg)
        pt_neg = torch.clamp(pt_neg, min=self.eps, max=1-self.eps)
        focal_weight_neg = pt_neg ** self.gamma_neg
        
        # BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 应用focal weight
        loss = focal_weight_pos * targets_pos * bce_loss + \
               focal_weight_neg * targets_neg * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    组合损失：Weighted BCE + Asymmetric Loss
    """
    def __init__(self, pos_weight=None, gamma_neg=4, gamma_pos=1, 
                 alpha=0.5, clip=0.05):
        """
        Args:
            pos_weight: 正样本权重（用于weighted BCE部分）
            gamma_neg: Asymmetric Loss的负样本gamma
            gamma_pos: Asymmetric Loss的正样本gamma
            alpha: 两个损失的权重（0-1之间）
            clip: Asymmetric Loss的clip参数
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.weighted_bce = WeightedBCELoss(pos_weight=pos_weight)
        self.asymmetric_loss = AsymmetricLoss(
            gamma_neg=gamma_neg, 
            gamma_pos=gamma_pos, 
            clip=clip
        )
    
    def forward(self, logits, targets):
        loss1 = self.weighted_bce(logits, targets)
        loss2 = self.asymmetric_loss(logits, targets)
        return self.alpha * loss1 + (1 - self.alpha) * loss2


def create_loss(loss_type='weighted_bce', pos_weight=None, 
                focal_alpha=0.25, focal_gamma=2.0,
                asymmetric_gamma_neg=4, asymmetric_gamma_pos=1,
                asymmetric_clip=0.05, combined_alpha=0.5):
    """
    创建损失函数的便捷函数
    
    Args:
        loss_type: 损失类型 ('weighted_bce', 'focal_loss', 'asymmetric_loss', 'combined')
        pos_weight: 正样本权重（用于weighted_bce）
        focal_alpha: Focal Loss的alpha参数
        focal_gamma: Focal Loss的gamma参数
        asymmetric_gamma_neg: Asymmetric Loss的负样本gamma
        asymmetric_gamma_pos: Asymmetric Loss的正样本gamma
        asymmetric_clip: Asymmetric Loss的clip参数
        combined_alpha: Combined Loss的权重
    """
    if loss_type == 'weighted_bce':
        return WeightedBCELoss(pos_weight=pos_weight)
    elif loss_type == 'focal_loss':
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    elif loss_type == 'asymmetric_loss':
        return AsymmetricLoss(
            gamma_neg=asymmetric_gamma_neg,
            gamma_pos=asymmetric_gamma_pos,
            clip=asymmetric_clip
        )
    elif loss_type == 'combined':
        return CombinedLoss(
            pos_weight=pos_weight,
            gamma_neg=asymmetric_gamma_neg,
            gamma_pos=asymmetric_gamma_pos,
            alpha=combined_alpha,
            clip=asymmetric_clip
        )
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




