"""
14类疾病标签定义与编码/解码工具
"""

# 14类疾病标签（不包括No Finding）
CLASS_NAMES = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Effusion',
    'Emphysema',
    'Fibrosis',
    'Hernia',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pleural_Thickening',
    'Pneumonia',
    'Pneumothorax'
]

NUM_CLASSES = len(CLASS_NAMES)


def labels_to_multihot(label_strings, class_names=None):
    """
    将标签字符串转换为multi-hot向量
    
    Args:
        label_strings: 标签字符串列表，例如 ["Cardiomegaly|Effusion", "No Finding", ...]
        class_names: 类别名称列表，默认使用 CLASS_NAMES
    
    Returns:
        multihot: numpy数组，形状为 (len(label_strings), num_classes)
    """
    import numpy as np
    import pandas as pd
    
    if class_names is None:
        class_names = CLASS_NAMES
    
    num_classes = len(class_names)
    multihot = np.zeros((len(label_strings), num_classes), dtype=np.float32)
    
    for idx, label_str in enumerate(label_strings):
        if pd.isna(label_str) or str(label_str).strip() == '':
            continue
        
        labels = [l.strip() for l in str(label_str).split('|')]
        
        for label in labels:
            if label == 'No Finding':
                continue
            if label in class_names:
                class_idx = class_names.index(label)
                multihot[idx, class_idx] = 1.0
    
    return multihot


def multihot_to_labels(multihot, class_names=None, threshold=0.5):
    """
    将multi-hot向量转换为标签字符串
    
    Args:
        multihot: numpy数组，形状为 (num_samples, num_classes) 或 (num_classes,)
        class_names: 类别名称列表，默认使用 CLASS_NAMES
        threshold: 阈值，超过此值视为正样本
    
    Returns:
        label_strings: 标签字符串列表
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    if len(multihot.shape) == 1:
        multihot = multihot.reshape(1, -1)
    
    label_strings = []
    for i in range(multihot.shape[0]):
        labels = []
        for j, class_name in enumerate(class_names):
            if multihot[i, j] > threshold:
                labels.append(class_name)
        
        if len(labels) == 0:
            label_strings.append('No Finding')
        else:
            label_strings.append('|'.join(labels))
    
    return label_strings


if __name__ == "__main__":
    # 测试
    test_labels = ["Cardiomegaly|Effusion", "No Finding", "Pneumonia"]
    multihot = labels_to_multihot(test_labels)
    print("Multi-hot编码:")
    print(multihot)
    
    recovered = multihot_to_labels(multihot)
    print("\n恢复的标签:")
    print(recovered)

