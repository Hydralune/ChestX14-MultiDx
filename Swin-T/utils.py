"""
工具函数模块
包含：随机种子设置、checkpoint保存/加载、logger、配置加载等
"""

import os
import yaml
import torch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter


def set_seed(seed=42):
    """
    设置随机种子，确保可复现性
    
    Args:
        seed: 随机种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        config: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(checkpoint_dir, epoch, model, optimizer, scheduler=None, 
                   scaler=None, best_auc=0.0, train_losses=None, 
                   val_losses=None, val_aucs=None, config=None, is_best=False):
    """
    保存检查点
    
    Args:
        checkpoint_dir: 检查点保存目录
        epoch: 当前epoch
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器（可选）
        scaler: 混合精度scaler（可选）
        best_auc: 最佳AUC
        train_losses: 训练损失历史
        val_losses: 验证损失历史
        val_aucs: 验证AUC历史
        config: 配置字典
        is_best: 是否为最佳模型
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_auc': best_auc,
        'train_losses': train_losses or [],
        'val_losses': val_losses or [],
        'val_aucs': val_aucs or [],
        'config': config
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    # 保存最新检查点
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, latest_path)
    
    # 保存最佳模型
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
        print(f"保存最佳模型 (AUC: {best_auc:.4f})")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
    """
    加载检查点
    
    Args:
        checkpoint_path: 检查点路径
        model: 模型
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        scaler: 混合精度scaler（可选）
    
    Returns:
        checkpoint: 检查点字典
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载调度器状态
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # 加载scaler状态
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint


class AverageMeter:
    """
    计算和存储平均值和当前值
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_logger(log_dir):
    """
    创建TensorBoard logger
    
    Args:
        log_dir: 日志目录
    
    Returns:
        writer: SummaryWriter
    """
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    return writer


if __name__ == "__main__":
    # 测试
    print("测试工具函数...")
    
    # 测试随机种子
    set_seed(42)
    print("随机种子设置成功")
    
    # 测试配置加载
    try:
        config = load_config('config.yaml')
        print(f"配置加载成功: {config['model']['name']}")
    except:
        print("配置加载失败（可能文件不存在）")
    
    # 测试AverageMeter
    meter = AverageMeter()
    meter.update(1.0, 10)
    meter.update(2.0, 20)
    print(f"AverageMeter测试: avg={meter.avg:.2f}, count={meter.count}")
    
    print("测试完成！")

