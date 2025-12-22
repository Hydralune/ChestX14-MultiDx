"""
评估指标计算模块
支持AUC (macro/micro) 和 F1 (macro/micro)，以及每类AUC
"""

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, auc


def calculate_metrics(labels, probs, threshold=0.5):
    """
    计算整体评估指标
    
    Args:
        labels: 真实标签 (N, num_classes)
        probs: 预测概率 (N, num_classes)
        threshold: 二分类阈值
    
    Returns:
        metrics: 字典，包含各种指标
    """
    # AUC
    try:
        auc_macro = roc_auc_score(labels, probs, average='macro')
    except:
        auc_macro = 0.0
    
    try:
        auc_micro = roc_auc_score(labels, probs, average='micro')
    except:
        auc_micro = 0.0
    
    # F1
    preds = (probs > threshold).astype(int)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    
    return {
        'auc_macro': auc_macro,
        'auc_micro': auc_micro,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro
    }


def calculate_per_class_auc(labels, probs, class_names):
    """
    计算每类的AUC
    
    Args:
        labels: 真实标签 (N, num_classes)
        probs: 预测概率 (N, num_classes)
        class_names: 类别名称列表
    
    Returns:
        class_aucs: 每类AUC列表
    """
    class_aucs = []
    for i, class_name in enumerate(class_names):
        try:
            class_auc = roc_auc_score(labels[:, i], probs[:, i])
            class_aucs.append(class_auc)
        except:
            # 如果没有正样本或负样本，AUC无法计算
            class_aucs.append(0.0)
    
    return class_aucs


def calculate_roc_curves(labels, probs, class_names):
    """
    计算每类的ROC曲线数据
    
    Args:
        labels: 真实标签 (N, num_classes)
        probs: 预测概率 (N, num_classes)
        class_names: 类别名称列表
    
    Returns:
        roc_data: 列表，每个元素是 (fpr, tpr, auc) 的元组
    """
    roc_data = []
    for i, class_name in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(labels[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            roc_data.append((fpr, tpr, roc_auc))
        except:
            roc_data.append((np.array([0, 1]), np.array([0, 1]), 0.0))
    
    return roc_data


if __name__ == "__main__":
    # 测试
    from labels import CLASS_NAMES
    
    # 创建随机数据
    n_samples = 100
    n_classes = len(CLASS_NAMES)
    
    labels = np.random.randint(0, 2, (n_samples, n_classes)).astype(float)
    probs = np.random.rand(n_samples, n_classes)
    
    # 计算指标
    metrics = calculate_metrics(labels, probs)
    print("整体指标:")
    print(f"  AUC (Macro): {metrics['auc_macro']:.4f}")
    print(f"  AUC (Micro): {metrics['auc_micro']:.4f}")
    print(f"  F1 (Macro): {metrics['f1_macro']:.4f}")
    print(f"  F1 (Micro): {metrics['f1_micro']:.4f}")
    
    # 计算每类AUC
    class_aucs = calculate_per_class_auc(labels, probs, CLASS_NAMES)
    print("\n每类AUC:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name:20s}: {class_aucs[i]:.4f}")

