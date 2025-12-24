"""
胸部X光多标签分类数据集
支持患者级划分、多标签转换、数据增强
注意：数据集在images文件夹，标签在filtered_labels.csv（相对于项目根目录）
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


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


class ChestXRayDataset(Dataset):
    def __init__(self, csv_path, image_dir, class_names, patient_ids,
                 transform=None, is_training=False):
        """
        Args:
            csv_path: CSV文件路径（相对于项目根目录，如 "../filtered_labels.csv"）
            image_dir: 图像目录路径（相对于项目根目录，如 "../images"）
            class_names: 类别名称列表
            patient_ids: 患者ID列表（用于过滤）
            transform: 图像变换
            is_training: 是否为训练模式
        """
        # 获取当前文件所在目录（convnext文件夹）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取项目根目录（上一级目录）
        project_root = os.path.dirname(current_dir)
        
        # 构建绝对路径
        self.csv_path = os.path.join(project_root, csv_path) if not os.path.isabs(csv_path) else csv_path
        self.image_dir = os.path.join(project_root, image_dir) if not os.path.isabs(image_dir) else image_dir
        
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.transform = transform
        self.is_training = is_training

        df = pd.read_csv(self.csv_path)

        # 过滤患者
        df = df[df['Patient ID'].isin(patient_ids)].reset_index(drop=True)

        # 只保留真实存在的图像
        existing_images = set(os.listdir(self.image_dir))

        valid_rows = []
        for _, row in df.iterrows():
            if row['Image Index'] in existing_images:
                valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        # 转换为multi-hot向量
        self.labels = self._labels_to_multihot(self.df['Finding Labels'].values)

        print(f"有效样本数: {len(self.df)} / 原始: {len(df)}")

    def _labels_to_multihot(self, label_strings):
        """
        将标签字符串转换为multi-hot向量
        例如: "Cardiomegaly|Effusion" -> [0,1,0,0,1,0,...]
        """
        multihot = np.zeros((len(label_strings), self.num_classes), dtype=np.float32)

        for idx, label_str in enumerate(label_strings):
            if pd.isna(label_str) or str(label_str).strip() == '':
                continue

            labels = [l.strip() for l in str(label_str).split('|')]

            for label in labels:
                if label == 'No Finding':
                    continue
                if label in self.class_names:
                    class_idx = self.class_names.index(label)
                    multihot[idx, class_idx] = 1.0

        return multihot

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.df.iloc[idx]['Image Index']
        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.FloatTensor(self.labels[idx])
        return image, label


def get_transforms(image_size=512, is_training=False, mean=None, std=None):
    """
    获取数据增强变换
    针对医学X光图像优化的数据增强策略
    """
    # === 医学X光图像归一化 ===
    # 将[0, 255]映射到[-1, 1]
    # 使用mean=0.5, std=0.5相当于 (x/255.0 - 0.5) / 0.5 = (x - 127.5) / 127.5
    if mean is None:
        mean = [0.5, 0.5, 0.5]  # 灰度图像的统一归一化值（3通道相同）
    if std is None:
        std = [0.5, 0.5, 0.5]   # 标准差0.5，实现[-1, 1]归一化
    
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            # 适度的几何变换（保留医学图像的空间结构）
            transforms.RandomRotation(degrees=10),  # 减小旋转角度，避免破坏病灶结构
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # 减小变换幅度
            # 保留轻微的亮度对比度调整（模拟不同设备的成像差异）
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 减小扰动幅度
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    return transform


def split_by_patient(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    按患者ID划分数据集，确保同一患者的所有图像在同一集合中
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    # 获取所有唯一的患者ID
    patient_ids = df['Patient ID'].unique()
    np.random.seed(random_state)
    np.random.shuffle(patient_ids)
    
    # 计算划分点
    n_patients = len(patient_ids)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    
    train_patients = patient_ids[:n_train]
    val_patients = patient_ids[n_train:n_train+n_val]
    test_patients = patient_ids[n_train+n_val:]
    
    print(f"患者划分 - 训练: {len(train_patients)}, 验证: {len(val_patients)}, 测试: {len(test_patients)}")
    
    return train_patients, val_patients, test_patients


def get_data_loaders(csv_path, image_dir, class_names, image_size=512, 
                     batch_size=16, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                     mean=None, std=None, num_workers=8, random_state=42):
    """
    获取数据加载器
    注意：csv_path和image_dir应该是相对于项目根目录的路径
    """
    # 获取当前文件所在目录（convnext文件夹）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录（上一级目录）
    project_root = os.path.dirname(current_dir)
    
    # 构建绝对路径
    csv_abs_path = os.path.join(project_root, csv_path) if not os.path.isabs(csv_path) else csv_path
    
    # 读取CSV
    df = pd.read_csv(csv_abs_path)
    
    # 按患者划分
    train_patients, val_patients, test_patients = split_by_patient(
        df, train_ratio, val_ratio, test_ratio, random_state
    )
    
    # 创建数据集
    train_dataset = ChestXRayDataset(
        csv_path, image_dir, class_names, train_patients,
        transform=get_transforms(image_size, is_training=True, mean=mean, std=std),
        is_training=True
    )
    
    val_dataset = ChestXRayDataset(
        csv_path, image_dir, class_names, val_patients,
        transform=get_transforms(image_size, is_training=False, mean=mean, std=std),
        is_training=False
    )
    
    test_dataset = ChestXRayDataset(
        csv_path, image_dir, class_names, test_patients,
        transform=get_transforms(image_size, is_training=False, mean=mean, std=std),
        is_training=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader, test_loader


def calculate_pos_weight(csv_path, class_names, patient_ids=None):
    """
    计算正样本权重，增加平滑处理防止权重过大
    """
    # 获取当前文件所在目录（convnext文件夹）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录（上一级目录）
    project_root = os.path.dirname(current_dir)
    
    # 构建绝对路径
    csv_abs_path = os.path.join(project_root, csv_path) if not os.path.isabs(csv_path) else csv_path
    
    df = pd.read_csv(csv_abs_path)

    if patient_ids is not None:
        df = df[df['Patient ID'].isin(patient_ids)]

    num_classes = len(class_names)
    labels = np.zeros((len(df), num_classes), dtype=np.float32)

    for i, label_str in enumerate(df['Finding Labels']):
        if pd.isna(label_str) or str(label_str).strip() == '':
            continue

        for label in str(label_str).split('|'):
            label = label.strip()
            if label == 'No Finding':
                continue
            if label in class_names:
                labels[i, class_names.index(label)] = 1.0

    num_positive = labels.sum(axis=0)
    num_negative = len(labels) - num_positive

    # 防止除零
    num_positive = np.maximum(num_positive, 1)
    num_negative = np.maximum(num_negative, 1)

    pos_weight = np.sqrt(num_negative / num_positive)
    
    # 转换为tensor并截断最大值，防止极度不平衡破坏梯度
    pos_weight = torch.FloatTensor(pos_weight)
    pos_weight = torch.clamp(pos_weight, max=10.0)

    return pos_weight


if __name__ == "__main__":
    # 测试数据集
    csv_path = "filtered_labels.csv"  # 相对于项目根目录
    image_dir = "images"  # 相对于项目根目录
    
    train_loader, val_loader, test_loader = get_data_loaders(
        csv_path, image_dir, CLASS_NAMES, image_size=512, batch_size=4
    )
    
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    
    # 测试一个batch
    images, labels = next(iter(train_loader))
    print(f"图像形状: {images.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"标签示例: {labels[0]}")

