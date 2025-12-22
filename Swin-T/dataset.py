"""
胸部X光多标签分类数据集
支持患者级划分、多标签转换、数据增强
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from labels import CLASS_NAMES, labels_to_multihot


class ChestXRayDataset(Dataset):
    def __init__(self, csv_path, image_dir, class_names, patient_ids,
                 transform=None, is_training=False):
        """
        Args:
            csv_path: CSV文件路径
            image_dir: 图像目录路径
            class_names: 类别名称列表
            patient_ids: 患者ID列表（用于过滤）
            transform: 图像变换
            is_training: 是否为训练模式
        """
        self.image_dir = image_dir
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.transform = transform
        self.is_training = is_training

        df = pd.read_csv(csv_path)

        # 过滤患者
        df = df[df['Patient ID'].isin(patient_ids)].reset_index(drop=True)

        # 只保留真实存在的图像
        existing_images = set(os.listdir(image_dir))

        valid_rows = []
        for _, row in df.iterrows():
            if row['Image Index'] in existing_images:
                valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        # 转换为multi-hot向量
        self.labels = labels_to_multihot(
            self.df['Finding Labels'].values, 
            class_names=class_names
        )

        print(f"有效样本数: {len(self.df)} / 原始: {len(df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.df.iloc[idx]['Image Index']
        image_path = os.path.join(self.image_dir, image_name)

        # 读取图像（X光图通常是灰度图）
        image = Image.open(image_path).convert('L')  # 转换为灰度图

        if self.transform:
            image = self.transform(image)

        label = torch.FloatTensor(self.labels[idx])
        return image, label


def get_transforms(image_size=512, is_training=False, mean=None, std=None, 
                   input_channel_mode='expand'):
    """
    获取数据增强变换
    
    Args:
        image_size: 图像尺寸
        is_training: 是否为训练模式
        mean: 归一化均值
        std: 归一化标准差
        input_channel_mode: 输入通道处理方式
            - "expand": 将灰度图扩展到3通道（复制3次）
            - "modify": 保持1通道（需要模型第一层适配）
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    transform_list = []
    
    # Resize到512x512（Swin默认224，这里适配到512）
    transform_list.append(transforms.Resize((image_size, image_size)))
    
    if is_training:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])
    
    # 转换为Tensor
    transform_list.append(transforms.ToTensor())
    
    # 处理通道数
    if input_channel_mode == 'expand':
        # 将1通道扩展到3通道（复制3次）
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    # 如果input_channel_mode == 'modify'，则保持1通道，归一化时使用单通道的mean/std
    
    # 归一化
    if input_channel_mode == 'expand':
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    else:
        # 单通道归一化（使用ImageNet的均值标准差）
        transform_list.append(transforms.Normalize(mean=[0.485], std=[0.229]))
    
    transform = transforms.Compose(transform_list)
    return transform


def split_by_patient(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    按患者ID划分数据集，确保同一患者的所有图像在同一集合中
    
    Args:
        df: DataFrame
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_state: 随机种子
    
    Returns:
        train_patients, val_patients, test_patients: 患者ID列表
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
                     mean=None, std=None, num_workers=4, random_state=42,
                     input_channel_mode='expand'):
    """
    获取数据加载器
    
    Args:
        csv_path: CSV文件路径
        image_dir: 图像目录路径
        class_names: 类别名称列表
        image_size: 图像尺寸
        batch_size: 批次大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        mean: 归一化均值
        std: 归一化标准差
        num_workers: 数据加载器工作进程数
        random_state: 随机种子
        input_channel_mode: 输入通道处理方式
    
    Returns:
        train_loader, val_loader, test_loader: 数据加载器
    """
    # 读取CSV
    df = pd.read_csv(csv_path)
    
    # 按患者划分
    train_patients, val_patients, test_patients = split_by_patient(
        df, train_ratio, val_ratio, test_ratio, random_state
    )
    
    # 创建数据集
    train_dataset = ChestXRayDataset(
        csv_path, image_dir, class_names, train_patients,
        transform=get_transforms(image_size, is_training=True, mean=mean, std=std, 
                                input_channel_mode=input_channel_mode),
        is_training=True
    )
    
    val_dataset = ChestXRayDataset(
        csv_path, image_dir, class_names, val_patients,
        transform=get_transforms(image_size, is_training=False, mean=mean, std=std,
                                input_channel_mode=input_channel_mode),
        is_training=False
    )
    
    test_dataset = ChestXRayDataset(
        csv_path, image_dir, class_names, test_patients,
        transform=get_transforms(image_size, is_training=False, mean=mean, std=std,
                                input_channel_mode=input_channel_mode),
        is_training=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def calculate_pos_weight(csv_path, class_names, patient_ids=None):
    """
    计算正样本权重，用于Weighted BCE Loss
    
    Args:
        csv_path: CSV文件路径
        class_names: 类别名称列表
        patient_ids: 患者ID列表（用于过滤）
    
    Returns:
        pos_weight: torch.Tensor，形状为 (num_classes,)
    """
    df = pd.read_csv(csv_path)

    if patient_ids is not None:
        df = df[df['Patient ID'].isin(patient_ids)]

    labels = labels_to_multihot(df['Finding Labels'].values, class_names=class_names)

    num_positive = labels.sum(axis=0)
    num_negative = len(labels) - num_positive

    # 防止除零
    num_positive = np.maximum(num_positive, 1)
    num_negative = np.maximum(num_negative, 1)

    # 使用平方根平滑权重
    pos_weight = np.sqrt(num_negative / num_positive)
    
    # 转换为tensor并截断最大值，防止极度不平衡破坏梯度
    pos_weight = torch.FloatTensor(pos_weight)
    pos_weight = torch.clamp(pos_weight, max=10.0)

    return pos_weight


if __name__ == "__main__":
    # 测试数据集
    csv_path = "../filtered_labels.csv"
    image_dir = "../images"
    
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

