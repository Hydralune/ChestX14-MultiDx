"""
ConvNeXt Large 训练脚本
支持4卡多GPU训练、Mixed Precision、EMA、Resume、自动保存best、TensorBoard
使用SOTA级别的训练方法
"""

import os
import yaml
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score

from dataset import get_data_loaders, CLASS_NAMES, calculate_pos_weight
from model import create_model
from loss import create_loss


class ModelEma(nn.Module):
    """
    模型指数移动平均 (Exponential Moving Average)
    这是一种免费的模型集成技术，通常能带来 0.5% - 1.0% 的 AUC 提升。
    """
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEma, self).__init__()
        # 深度拷贝原模型
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            # 处理DataParallel包装的模型
            if isinstance(model, nn.DataParallel):
                model_dict = model.module.state_dict()
            else:
                model_dict = model.state_dict()
                
            for ema_v, model_v in zip(self.module.state_dict().values(), model_dict.values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class Trainer:
    """训练器类 - 支持多GPU训练"""
    
    def __init__(self, config):
        self.config = config
        
        # 检测可用GPU数量
        self.num_gpus = torch.cuda.device_count()
        print(f"检测到 {self.num_gpus} 个GPU")
        
        if self.num_gpus > 0:
            self.device = torch.device('cuda:0')
            print(f"使用设备: {self.device}")
        else:
            self.device = torch.device('cpu')
            print("警告: 未检测到GPU，使用CPU训练（速度会很慢）")
        
        # 创建目录
        os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['paths']['log_dir'], exist_ok=True)
        os.makedirs(config['paths']['result_dir'], exist_ok=True)
        
        # 设置随机种子
        self._set_seed(config.get('seed', 42))
        
        # 初始化数据加载器
        self._init_dataloaders()
        
        # 初始化模型
        self._init_model()
        
        # 初始化 EMA 模型
        print("初始化 EMA 模型...")
        self.model_ema = ModelEma(self.model, decay=config['train'].get('ema_decay', 0.999))
        
        # 初始化损失函数
        self._init_loss()
        
        # 初始化优化器和调度器
        self._init_optimizer()
        
        # 初始化混合精度训练
        self.scaler = GradScaler() if config['train']['mixed_precision'] else None
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=config['paths']['log_dir'])
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.best_auc = 0.0
        self.start_epoch = 0
        
        # 早停机制参数
        self.patience = config['train'].get('patience', 8)
        self.counter = 0
        
        # Resume训练
        if config['train']['resume'] and config['train']['resume_checkpoint']:
            self._load_checkpoint(config['train']['resume_checkpoint'])
    
    def _set_seed(self, seed):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _init_dataloaders(self):
        """初始化数据加载器"""
        data_config = self.config['data']
        image_config = self.config['image']
        
        # 根据GPU数量调整batch size和workers
        batch_size = self.config['train']['batch_size']
        num_workers = self.config['train'].get('num_workers', 8)
        
        # 多GPU训练时，每个GPU的batch size会平均分配
        print(f"总batch size: {batch_size}, GPU数量: {self.num_gpus}")
        if self.num_gpus > 1:
            print(f"每个GPU的batch size: {batch_size // self.num_gpus}")
        
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(
            csv_path=data_config['csv_path'],
            image_dir=data_config['image_dir'],
            class_names=CLASS_NAMES,
            image_size=image_config['size'],
            batch_size=batch_size,
            train_ratio=data_config['train_ratio'],
            val_ratio=data_config['val_ratio'],
            test_ratio=data_config['test_ratio'],
            mean=image_config['mean'],
            std=image_config['std'],
            num_workers=num_workers,
            random_state=self.config.get('seed', 42)
        )
        
        # 计算pos_weight（仅使用训练集）
        print("计算正样本权重...")
        train_patients = self.train_loader.dataset.df['Patient ID'].unique()
        self.pos_weight = calculate_pos_weight(
            data_config['csv_path'], CLASS_NAMES, train_patients
        ).to(self.device)
        print(f"正样本权重: {self.pos_weight.cpu().numpy()}")
    
    def _init_model(self):
        """初始化模型并支持多GPU"""
        model_config = self.config['model']
        print(f"正在创建模型: {model_config['name']} (预训练: {model_config['pretrained']})")
        self.model = create_model(
            model_name=model_config['name'],
            num_classes=model_config['num_classes'],
            pretrained=model_config['pretrained']
        )
        self.model = self.model.to(self.device)
        
        # 多GPU训练
        if self.num_gpus > 1:
            print(f"使用 {self.num_gpus} 个GPU进行训练 (DataParallel)")
            self.model = nn.DataParallel(self.model)
        
        # 计算参数量
        if isinstance(self.model, nn.DataParallel):
            model_for_params = self.model.module
        else:
            model_for_params = self.model
            
        total_params = sum(p.numel() for p in model_for_params.parameters())
        trainable_params = sum(p.numel() for p in model_for_params.parameters() if p.requires_grad)
        print(f"模型参数量 - 总数: {total_params:,}, 可训练: {trainable_params:,}")
    
    def _init_loss(self):
        """初始化损失函数（SOTA版本）"""
        train_config = self.config['train']
        loss_type = train_config.get('loss_type', 'asymmetric_loss')  # 默认使用ASL
        
        if loss_type == 'weighted_bce':
            self.criterion = create_loss(
                loss_type='weighted_bce',
                pos_weight=self.pos_weight
            )
        elif loss_type == 'focal_loss':
            self.criterion = create_loss(
                loss_type='focal_loss',
                focal_alpha=train_config.get('focal_alpha', 0.25),
                focal_gamma=train_config.get('focal_gamma', 2.0)
            )
        elif loss_type == 'asymmetric_loss':
            # ASL Loss (SOTA损失函数，适合多标签医学影像)
            self.criterion = create_loss(
                loss_type='asymmetric_loss',
                asl_gamma_pos=train_config.get('asl_gamma_pos', 1.0),
                asl_gamma_neg=train_config.get('asl_gamma_neg', 4.0),
                asl_clip=train_config.get('asl_clip', 0.05)
            )
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}")
        print(f"损失函数: {loss_type} (SOTA: Asymmetric Loss推荐用于多标签医学影像)")
    
    def _init_optimizer(self):
        """初始化优化器和调度器"""
        train_config = self.config['train']
        
        # 优化器 - 使用AdamW (SOTA优化器)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学习率调度器
        if train_config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_config['num_epochs'],
                eta_min=train_config.get('min_lr', 1e-6)
            )
        elif train_config['scheduler'] == 'cosine_warmup':
            # Cosine Annealing with Warmup
            from torch.optim.lr_scheduler import LambdaLR
            warmup_epochs = train_config.get('warmup_epochs', 5)
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return epoch / max(warmup_epochs, 1)
                else:
                    progress = (epoch - warmup_epochs) / max(train_config['num_epochs'] - warmup_epochs, 1)
                    return 0.5 * (1 + np.cos(np.pi * progress))
            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        elif train_config['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=train_config['num_epochs'] // 3,
                gamma=0.1
            )
        else:
            self.scheduler = None
        
        print(f"优化器: AdamW, LR: {train_config['learning_rate']}, Scheduler: {train_config['scheduler']}")
    
    def train_epoch(self, epoch):
        """
        训练一个epoch
        
        【SOTA改进 - 移除Mixup】
        - Mixup会破坏医学图像中的局部病灶结构（如小病灶、边界清晰的病变）
        - 对AUC（排序指标）不利，因为混合后的标签是soft的，破坏了hard label的排序信息
        - 预期提升: +0.01~0.02 AUC（通过保留真实的病灶结构）
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["train"]["num_epochs"]} [Train]')
        
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # 混合精度前向传播
            if self.scaler is not None:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪 (防止新架构梯度爆炸)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()
            
            # 更新 EMA 模型
            if isinstance(self.model, nn.DataParallel):
                self.model_ema.update(self.model.module)
            else:
                self.model_ema.update(self.model)
            
            running_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = running_loss / num_batches
        return avg_loss
    
    def validate(self, epoch, use_ema=True):
        """验证 (支持使用 EMA 模型验证)"""
        # 如果使用 EMA，则评估 EMA 模型，否则评估当前模型
        eval_model = self.model_ema.module if use_ema else (self.model.module if isinstance(self.model, nn.DataParallel) else self.model)
        eval_model.eval()
        
        running_loss = 0.0
        all_probs = []
        all_labels = []
        
        mode_str = "[Val (EMA)]" if use_ema else "[Val]"
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} {mode_str}')
            
            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                if self.scaler is not None:
                    with autocast():
                        logits = eval_model(images)
                        loss = self.criterion(logits, labels)
                else:
                    logits = eval_model(images)
                    loss = self.criterion(logits, labels)
                
                running_loss += loss.item()
                
                # 计算概率
                probs = torch.sigmoid(logits).cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                all_probs.append(probs)
                all_labels.append(labels_np)
        
        avg_loss = running_loss / len(self.val_loader)
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        try:
            auc_macro = roc_auc_score(all_labels, all_probs, average='macro')
            auc_micro = roc_auc_score(all_labels, all_probs, average='micro')
        except:
            auc_macro = 0.0
            auc_micro = 0.0
        
        preds = (all_probs > 0.5).astype(int)
        f1_macro = f1_score(all_labels, preds, average='macro', zero_division=0)
        f1_micro = f1_score(all_labels, preds, average='micro', zero_division=0)
        
        return avg_loss, auc_macro, auc_micro, f1_macro, f1_micro
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点 (同时保存 EMA 状态)"""
        # 获取实际模型（去除DataParallel包装）
        if isinstance(self.model, nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'model_ema_state_dict': self.model_ema.module.state_dict(),  # 保存 EMA
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_auc': self.best_auc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_aucs': self.val_aucs,
            'config': self.config
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(
            self.config['paths']['checkpoint_dir'],
            'checkpoint_latest.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(
                self.config['paths']['checkpoint_dir'],
                'checkpoint_best.pth'
            )
            torch.save(checkpoint, best_path)
            # 同时也保存一个 EMA 版本的最佳权重（通常这是最好的）
            best_ema_path = os.path.join(
                self.config['paths']['checkpoint_dir'],
                'checkpoint_best_ema.pth'
            )
            torch.save({
                'model_state_dict': self.model_ema.module.state_dict(),
                'config': self.config,
                'epoch': epoch,
                'auc': self.best_auc
            }, best_ema_path)
            
            print(f"保存最佳模型 (AUC: {self.best_auc:.4f}) [EMA 版本已保存]")
    
    def _load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型权重
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 尝试加载 EMA
        if 'model_ema_state_dict' in checkpoint:
            self.model_ema.module.load_state_dict(checkpoint['model_ema_state_dict'])
            print("EMA 状态已恢复")
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_auc = checkpoint['best_auc']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_aucs = checkpoint['val_aucs']
        
        print(f"从epoch {self.start_epoch}恢复训练")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        axes[0].plot(self.train_losses, label='Train Loss', marker='o')
        axes[0].plot(self.val_losses, label='Val Loss (EMA)', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # AUC曲线
        axes[1].plot(self.val_aucs, label='Val AUC (EMA Macro)', marker='o', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].set_title('Validation AUC')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        # 保存图像
        curve_path = os.path.join(self.config['paths']['result_dir'], 'training_curves.png')
        plt.savefig(curve_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存: {curve_path}")
        plt.close()
    
    def train(self):
        """主训练循环"""
        num_epochs = self.config['train']['num_epochs']
        
        print("=" * 60)
        print("开始训练 ConvNeXt Large...")
        print(f"总epochs: {num_epochs}, 从epoch {self.start_epoch}开始")
        print(f"GPU数量: {self.num_gpus}")
        print("已启用: Model EMA, 梯度裁剪, 早停机制, 混合精度训练")
        print("=" * 60)
        
        for epoch in range(self.start_epoch, num_epochs):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证 (使用 EMA 模型，通常性能更好)
            val_loss, auc_macro, auc_micro, f1_macro, f1_micro = self.validate(epoch, use_ema=True)
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            # 记录
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_aucs.append(auc_macro)
            
            # TensorBoard记录
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val_EMA', val_loss, epoch)
            self.writer.add_scalar('AUC/Val_Macro_EMA', auc_macro, epoch)
            self.writer.add_scalar('AUC/Val_Micro_EMA', auc_micro, epoch)
            self.writer.add_scalar('F1/Val_Macro_EMA', f1_macro, epoch)
            self.writer.add_scalar('F1/Val_Micro_EMA', f1_micro, epoch)
            self.writer.add_scalar('LR', current_lr, epoch)
            
            # 打印信息
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Val Loss (EMA): {val_loss:.4f}")
            print(f"  Val AUC (Macro EMA): {auc_macro:.4f}, Val AUC (Micro EMA): {auc_micro:.4f}")
            print(f"  Val F1 (Macro EMA): {f1_macro:.4f}, Val F1 (Micro EMA): {f1_micro:.4f}")
            print(f"  LR: {current_lr:.6f}")
            print("-" * 60)
            
            # 保存检查点
            is_best = auc_macro > self.best_auc
            if is_best:
                self.best_auc = auc_macro
                self.counter = 0  # 重置早停计数器
            else:
                self.counter += 1
                print(f"早停计数: {self.counter}/{self.patience}")
            
            self.save_checkpoint(epoch, is_best=is_best)
            
            # 早停
            if self.counter >= self.patience:
                print(f"验证集性能在 {self.patience} 个 epoch 内未提升，停止训练。")
                break
        
        # 绘制训练曲线
        self.plot_training_curves()
        self.writer.close()
        print(f"训练完成! 最佳AUC: {self.best_auc:.4f}")


def main():
    """主函数"""
    # 加载配置文件
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()

