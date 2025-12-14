"""
训练脚本 (High Performance Version)
集成 Mixup、Model EMA、梯度裁剪、早停机制
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
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class Trainer:
    """训练器类"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
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
        
        # 初始化 EMA 模型 (新增)
        print("初始化 EMA 模型...")
        self.model_ema = ModelEma(self.model, decay=0.999)
        
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
        
        # 早停机制参数 (新增)
        self.patience = 8  # 如果8个epoch没提升就停止
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
        
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(
            csv_path=data_config['csv_path'],
            image_dir=data_config['image_dir'],
            class_names=CLASS_NAMES,
            image_size=image_config['size'],
            batch_size=self.config['train']['batch_size'],
            train_ratio=data_config['train_ratio'],
            val_ratio=data_config['val_ratio'],
            test_ratio=data_config['test_ratio'],
            mean=image_config['mean'],
            std=image_config['std'],
            random_state=self.config.get('seed', 42)
        )
        
        # 计算pos_weight（仅使用训练集）
        print("计算正样本权重...")
        train_patients = self.train_loader.dataset.df['Patient ID'].unique()
        self.pos_weight = calculate_pos_weight(
            data_config['csv_path'], CLASS_NAMES, train_patients
        ).to(self.device)
        print(f"正样本权重: {self.pos_weight}")
    
    def _init_model(self):
        """初始化模型"""
        model_config = self.config['model']
        print(f"正在创建模型: {model_config['name']} (预训练: {model_config['pretrained']})")
        self.model = create_model(
            model_name=model_config['name'],
            num_classes=model_config['num_classes'],
            pretrained=model_config['pretrained']
        )
        self.model = self.model.to(self.device)
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"模型参数量 - 总数: {total_params:,}, 可训练: {trainable_params:,}")
    
    def _init_loss(self):
        """初始化损失函数"""
        train_config = self.config['train']
        
        if train_config['loss_type'] == 'weighted_bce':
            self.criterion = create_loss(
                loss_type='weighted_bce',
                pos_weight=self.pos_weight
            )
        else:
            self.criterion = create_loss(
                loss_type='focal_loss',
                focal_alpha=train_config['focal_alpha'],
                focal_gamma=train_config['focal_gamma']
            )
        print(f"损失函数: {train_config['loss_type']}")
    
    def _init_optimizer(self):
        """初始化优化器和调度器"""
        train_config = self.config['train']
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
        
        # 学习率调度器
        if train_config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_config['num_epochs'],
                eta_min=1e-6
            )
        elif train_config['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=train_config['num_epochs'] // 3,
                gamma=0.1
            )
        else:
            self.scheduler = None
    
    def train_epoch(self, epoch):
        """训练一个epoch (集成 Mixup)"""
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["train"]["num_epochs"]} [Train]')
        
        # Mixup 参数
        mixup_alpha = 0.2
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # === Mixup 逻辑开始 ===
            use_mixup = False
            if np.random.random() < 0.5: # 50% 概率使用 Mixup
                use_mixup = True
                # 生成 Beta 分布的 lambda
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                # 生成打乱的索引
                index = torch.randperm(images.size(0)).to(self.device)
                
                # 混合图像
                mixed_images = lam * images + (1 - lam) * images[index]
                # 混合标签
                mixed_labels = lam * labels + (1 - lam) * labels[index]
            # === Mixup 逻辑结束 ===
            
            # 混合精度前向传播
            if self.scaler is not None:
                with autocast():
                    if use_mixup:
                        logits = self.model(mixed_images)
                        loss = self.criterion(logits, mixed_labels)
                    else:
                        logits = self.model(images)
                        loss = self.criterion(logits, labels)
                
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪 (防止新架构梯度爆炸)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if use_mixup:
                    logits = self.model(mixed_images)
                    loss = self.criterion(logits, mixed_labels)
                else:
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()
            
            # 更新 EMA 模型
            self.model_ema.update(self.model)
            
            running_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = running_loss / num_batches
        return avg_loss
    
    def validate(self, epoch, use_ema=True):
        """验证 (支持使用 EMA 模型验证)"""
        # 如果使用 EMA，则评估 EMA 模型，否则评估当前模型
        eval_model = self.model_ema.module if use_ema else self.model
        eval_model.eval()
        
        running_loss = 0.0
        all_probs = []
        all_labels = []
        
        mode_str = "[Val (EMA)]" if use_ema else "[Val]"
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} {mode_str}')
            
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
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
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'model_ema_state_dict': self.model_ema.module.state_dict(), # 保存 EMA
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
        
        print("开始训练...")
        print(f"总epochs: {num_epochs}, 从epoch {self.start_epoch}开始")
        print("已启用: Mixup 数据增强, Model EMA, 梯度裁剪, 早停机制")
        
        for epoch in range(self.start_epoch, num_epochs):
            # 训练 (带 Mixup)
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
            self.writer.add_scalar('LR', current_lr, epoch)
            
            # 打印信息
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Val Loss (EMA): {val_loss:.4f}")
            print(f"  Val AUC (Macro EMA): {auc_macro:.4f}")
            print(f"  LR: {current_lr:.6f}")
            print("-" * 60)
            
            # 保存检查点
            is_best = auc_macro > self.best_auc
            if is_best:
                self.best_auc = auc_macro
                self.counter = 0 # 重置早停计数器
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
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()