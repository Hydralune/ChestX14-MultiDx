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
    """胸部X光多标签分类数据集"""
    
    def __init__(self, csv_path, image_dir, class_names, patient_ids, 
                 transform=None, is_training=False):
        """
        Args:
            csv_path: CSV文件路径
            image_dir: 图像目录
            class_names: 类别名称列表
            patient_ids: 包含的患者ID列表
            transform: 图像变换
            is_training: 是否为训练集（用于数据增强）
        """
        self.image_dir = image_dir
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.transform = transform
        self.is_training = is_training
        
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        # 过滤指定患者的数据
        self.df = df[df['Patient ID'].isin(patient_ids)].reset_index(drop=True)
        
        # 转换为多标签格式
        self.labels = self._labels_to_multihot(self.df['Finding Labels'].values)
        
        print(f"数据集大小: {len(self.df)}, 患者数: {len(patient_ids)}")
        
    def _labels_to_multihot(self, label_strings):
        """
        将标签字符串转换为multi-hot向量
        例如: "Cardiomegaly|Effusion" -> [0,1,0,0,1,0,...]
        """
        multihot = np.zeros((len(label_strings), self.num_classes), dtype=np.float32)
        
        for idx, label_str in enumerate(label_strings):
            if pd.isna(label_str) or label_str.strip() == '':
                continue
            
            labels = [l.strip() for l in str(label_str).split('|')]
            
            for label in labels:
                if label == 'No Finding':
                    # No Finding表示无疾病，所有标签为0
                    continue
                if label in self.class_names:
                    class_idx = self.class_names.index(label)
                    multihot[idx, class_idx] = 1.0
        
        return multihot
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 获取图像路径
        image_name = self.df.iloc[idx]['Image Index']
        image_path = os.path.join(self.image_dir, image_name)
        
        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"加载图像失败: {image_path}, 错误: {e}")
            # 返回黑色图像
            image = Image.new('RGB', (512, 512), (0, 0, 0))
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 获取标签
        label = torch.FloatTensor(self.labels[idx])
        
        return image, label, image_name


def get_transforms(image_size=512, is_training=False, mean=None, std=None):
    """
    获取数据增强变换
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
                     mean=None, std=None, num_workers=4, random_state=42):
    """
    获取数据加载器
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
    计算正样本权重，用于处理类别不平衡
    """
    df = pd.read_csv(csv_path)
    
    if patient_ids is not None:
        df = df[df['Patient ID'].isin(patient_ids)]
    
    # 转换为多标签
    dataset = ChestXRayDataset(csv_path, "", class_names, 
                              df['Patient ID'].unique() if patient_ids is None else patient_ids,
                              transform=None)
    
    labels = dataset.labels
    num_positive = labels.sum(axis=0)
    num_negative = len(labels) - num_positive
    
    # 避免除零
    num_negative = np.maximum(num_negative, 1)
    num_positive = np.maximum(num_positive, 1)
    
    pos_weight = num_negative / num_positive
    
    return torch.FloatTensor(pos_weight)


if __name__ == "__main__":
    # 测试数据集
    csv_path = "filtered_labels.csv"
    image_dir = "images"
    
    train_loader, val_loader, test_loader = get_data_loaders(
        csv_path, image_dir, CLASS_NAMES, image_size=512, batch_size=4
    )
    
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    
    # 测试一个batch
    images, labels, names = next(iter(train_loader))
    print(f"图像形状: {images.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"标签示例: {labels[0]}")

