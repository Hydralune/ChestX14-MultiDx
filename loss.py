"""

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
    Asymmetric Loss (ASL) - 多标签分类的SOTA损失函数
    
    论文: "Asymmetric Loss For Multi-Label Classification"
    
    关键特性:
    1. 对positive和negative使用不同的gamma参数
    2. 通过margin机制抑制easy negative的梯度
    3. 特别适合弱标注、类别不平衡的医学影像任务
    
    ASL公式:
    - Positive: FL_pos = -alpha * (1 - p)^gamma_pos * log(p)
    - Negative: FL_neg = -alpha * (p)^gamma_neg * log(1-p) if p < margin else 0
    
    预期提升: +0.01~0.03 Macro AUC
    """
    
    def __init__(self, gamma_pos=1.0, gamma_neg=4.0, clip=0.05, 
                 alpha=1.0, reduction='mean', eps=1e-8):
        """
        Args:
            gamma_pos: positive样本的focal参数（通常较小，如1.0）
            gamma_neg: negative样本的focal参数（通常较大，如4.0，用于抑制easy negative）
            clip: negative样本的概率阈值，低于此值则抑制梯度
            alpha: 类别平衡因子（通常为1.0，可以进一步优化）
            reduction: 'mean' 或 'sum'
            eps: 数值稳定性参数
        """
        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.alpha = alpha
        self.reduction = reduction
        self.eps = eps
    
    def forward(self, logits, targets):
        """
        Args:
            logits: 模型输出 (B, num_classes) - 未经过sigmoid的logits
            targets: 真实标签 (B, num_classes) - 0或1
        Returns:
            loss: 损失值
        """
        # 计算概率
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        # ASL clipping: increase negative probability to suppress easy negatives
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # log-prob (numerically stable)
        log_pos = torch.log(xs_pos.clamp(min=self.eps))
        log_neg = torch.log(xs_neg.clamp(min=self.eps))

        # base CE for multi-label
        loss = targets * log_pos + (1.0 - targets) * log_neg

        # asymmetric focusing
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt = xs_pos * targets + xs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss = loss * torch.pow(1.0 - pt, gamma)

        # optional alpha balancing
        if self.alpha != 1.0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = loss * alpha_t

        loss = -loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def create_loss(loss_type='asymmetric_loss', pos_weight=None, 
                focal_alpha=0.25, focal_gamma=2.0,
                asl_gamma_pos=1.0, asl_gamma_neg=4.0, asl_clip=0.05):
    """
    创建损失函数的便捷函数
    
    Args:
        loss_type: 损失类型 ('weighted_bce', 'focal_loss', 或 'asymmetric_loss')
        pos_weight: 正样本权重（用于weighted_bce）
        focal_alpha: Focal Loss的alpha参数
        focal_gamma: Focal Loss的gamma参数
        asl_gamma_pos: ASL的positive gamma参数
        asl_gamma_neg: ASL的negative gamma参数
        asl_clip: ASL的negative clip阈值
    """
    if loss_type == 'weighted_bce':
        return WeightedBCELoss(pos_weight=pos_weight)
    elif loss_type == 'focal_loss':
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    elif loss_type == 'asymmetric_loss':
        return AsymmetricLoss(
            gamma_pos=asl_gamma_pos,
            gamma_neg=asl_gamma_neg,
            clip=asl_clip,
            alpha=1.0,
            reduction='mean'
        )
    else:
        raise ValueError(f"不支持的损失类型: {loss_type}。支持: 'weighted_bce', 'focal_loss', 'asymmetric_loss'")


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
    
    # 测试Asymmetric Loss (ASL) - SOTA损失函数
    asl_loss = AsymmetricLoss(gamma_pos=1.0, gamma_neg=4.0, clip=0.05)
    loss3 = asl_loss(logits, targets)
    print(f"Asymmetric Loss (ASL): {loss3.item():.4f}")


