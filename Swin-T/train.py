"""
训练脚本 - Swin Transformer
支持 Mixed Precision、Resume、自动保存 best、TensorBoard
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.amp import autocast as autocast_new
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
import copy

from dataset import get_data_loaders, CLASS_NAMES, calculate_pos_weight, mixup_data, mixup_criterion
from model import create_model
from loss import create_loss
from metrics import calculate_metrics
from utils import set_seed, load_config, save_checkpoint, load_checkpoint, create_logger


class ModelEMA:
    """指数移动平均模型，用于提升模型性能"""
    def __init__(self, model, decay=0.9999, device=None):
        """
        Args:
            model: 要平滑的模型
            decay: EMA衰减率
            device: 设备
        """
        self.model = model
        self.decay = decay
        self.device = device
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """注册模型参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """更新EMA参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """应用EMA参数到模型"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """恢复原始参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


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
        set_seed(config.get('seed', 42))
        
        # 初始化数据加载器
        self._init_dataloaders()
        
        # 初始化模型
        self._init_model()
        
        # 初始化损失函数
        self._init_loss()
        
        # 初始化优化器和调度器
        self._init_optimizer()
        
        # 初始化EMA
        ema_decay = config['train'].get('ema_decay', 0.9999)
        self.use_ema = config['train'].get('use_ema', True)
        if self.use_ema:
            self.ema = ModelEMA(self.model, decay=ema_decay, device=self.device)
            print(f"启用EMA，衰减率: {ema_decay}")
        else:
            self.ema = None
        
        # 初始化混合精度训练
        self.scaler = GradScaler() if config['train']['mixed_precision'] else None
        
        # 梯度累积步数
        self.accumulation_steps = config['train'].get('gradient_accumulation_steps', 1)
        
        # Mixup配置
        self.use_mixup = config['train'].get('use_mixup', False)
        self.mixup_alpha = config['train'].get('mixup_alpha', 0.2)
        if self.use_mixup:
            print(f"启用Mixup数据增强，alpha: {self.mixup_alpha}")
        
        # TensorBoard
        self.writer = create_logger(config['paths']['log_dir'])
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.best_auc = 0.0
        self.start_epoch = 0
        
        # Resume训练
        if config['train']['resume'] and config['train']['resume_checkpoint']:
            self._load_checkpoint(config['train']['resume_checkpoint'])
    
    def _init_dataloaders(self):
        """初始化数据加载器"""
        data_config = self.config['data']
        image_config = self.config['image']
        model_config = self.config['model']
        
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
            random_state=self.config.get('seed', 42),
            input_channel_mode=model_config.get('input_channel_mode', 'expand')
        )
        
        # 计算pos_weight（仅使用训练集）
        print("计算正样本权重...")
        train_patients = self.train_loader.dataset.df['Patient ID'].unique()
        self.pos_weight = calculate_pos_weight(
            data_config['csv_path'], CLASS_NAMES, train_patients
        ).to(self.device)
        print(f"正样本权重: {self.pos_weight.cpu().numpy()}")
    
    def _init_model(self):
        """初始化模型"""
        model_config = self.config['model']
        image_config = self.config['image']
        print(f"正在创建模型: {model_config['name']} (预训练: {model_config['pretrained']})")
        print(f"输入通道模式: {model_config.get('input_channel_mode', 'expand')}")
        
        self.model = create_model(
            num_classes=model_config['num_classes'],
            pretrained=model_config['pretrained'],
            input_channel_mode=model_config.get('input_channel_mode', 'expand'),
            img_size=image_config['size'],
            model_size=model_config.get('model_size', 'tiny')
        )
        self.model = self.model.to(self.device)
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"模型参数量 - 总数: {total_params:,}, 可训练: {trainable_params:,}")
    
    def _init_loss(self):
        """初始化损失函数"""
        train_config = self.config['train']
        loss_type = train_config['loss_type']
        
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
            self.criterion = create_loss(
                loss_type='asymmetric_loss',
                asymmetric_gamma_neg=train_config.get('asymmetric_gamma_neg', 4),
                asymmetric_gamma_pos=train_config.get('asymmetric_gamma_pos', 1),
                asymmetric_clip=train_config.get('asymmetric_clip', 0.05)
            )
        elif loss_type == 'combined':
            self.criterion = create_loss(
                loss_type='combined',
                pos_weight=self.pos_weight,
                asymmetric_gamma_neg=train_config.get('asymmetric_gamma_neg', 4),
                asymmetric_gamma_pos=train_config.get('asymmetric_gamma_pos', 1),
                asymmetric_clip=train_config.get('asymmetric_clip', 0.05),
                combined_alpha=train_config.get('combined_alpha', 0.5)
            )
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}")
        print(f"损失函数: {loss_type}")
    
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
            # Cosine Annealing with Warmup
            warmup_epochs = train_config.get('warmup_epochs', 5)
            total_epochs = train_config['num_epochs']
            eta_min = train_config.get('eta_min', 1e-6)
            
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    # Warmup: 线性增长
                    return epoch / warmup_epochs
                else:
                    # Cosine annealing
                    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                    return eta_min / train_config['learning_rate'] + (1 - eta_min / train_config['learning_rate']) * 0.5 * (1 + np.cos(np.pi * progress))
            
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            print(f"使用Cosine Annealing调度器，Warmup: {warmup_epochs} epochs")
        elif train_config['scheduler'] == 'linear':
            # Linear warmup + linear decay
            warmup_epochs = train_config.get('warmup_epochs', 5)
            total_epochs = train_config['num_epochs']
            
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return epoch / warmup_epochs
                else:
                    return (total_epochs - epoch) / (total_epochs - warmup_epochs)
            
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            self.scheduler = None
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        
        # 梯度累积相关
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["train"]["num_epochs"]} [Train]')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Mixup数据增强
            if self.use_mixup and np.random.rand() < 0.5:  # 50%概率使用Mixup
                images, labels_a, labels_b, lam = mixup_data(images, labels, self.mixup_alpha)
                use_mixup = True
            else:
                labels_a, labels_b, lam = labels, labels, 1.0
                use_mixup = False
            
            # 混合精度前向传播
            if self.scaler is not None:
                # 使用新的autocast API避免警告
                with autocast_new('cuda', enabled=True):
                    logits = self.model(images)
                    if use_mixup:
                        loss = mixup_criterion(self.criterion, logits, labels_a, labels_b, lam)
                    else:
                        loss = self.criterion(logits, labels)
                    # 梯度累积：除以累积步数
                    loss = loss / self.accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                # 梯度累积：每accumulation_steps步更新一次
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    # 更新EMA
                    if self.ema is not None:
                        self.ema.update()
            else:
                with autocast_new('cuda', enabled=False):
                    logits = self.model(images)
                    if use_mixup:
                        loss = mixup_criterion(self.criterion, logits, labels_a, labels_b, lam)
                    else:
                        loss = self.criterion(logits, labels)
                    loss = loss / self.accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if self.ema is not None:
                        self.ema.update()
            
            running_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item() * self.accumulation_steps:.4f}'})
        
        # 处理最后一个不完整的batch
        if len(self.train_loader) % self.accumulation_steps != 0:
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            if self.ema is not None:
                self.ema.update()
        
        avg_loss = running_loss / num_batches
        return avg_loss
    
    def validate(self, epoch):
        """验证"""
        # 使用EMA模型进行验证
        if self.ema is not None:
            self.ema.apply_shadow()
        
        self.model.eval()
        
        running_loss = 0.0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
            
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                if self.scaler is not None:
                    with autocast_new('cuda', enabled=True):
                        logits = self.model(images)
                        loss = self.criterion(logits, labels)
                else:
                    with autocast_new('cuda', enabled=False):
                        logits = self.model(images)
                        loss = self.criterion(logits, labels)
                
                running_loss += loss.item()
                
                # 计算概率
                probs = torch.sigmoid(logits).cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                all_probs.append(probs)
                all_labels.append(labels_np)
        
        # 恢复原始模型参数
        if self.ema is not None:
            self.ema.restore()
        
        avg_loss = running_loss / len(self.val_loader)
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # 计算指标
        metrics = calculate_metrics(all_labels, all_probs)
        
        return avg_loss, metrics['auc_macro'], metrics['auc_micro'], \
               metrics['f1_macro'], metrics['f1_micro']
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        save_checkpoint(
            checkpoint_dir=self.config['paths']['checkpoint_dir'],
            epoch=epoch,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            best_auc=self.best_auc,
            train_losses=self.train_losses,
            val_losses=self.val_losses,
            val_aucs=self.val_aucs,
            config=self.config,
            is_best=is_best
        )
    
    def _load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = load_checkpoint(
            checkpoint_path, 
            self.model, 
            self.optimizer, 
            self.scheduler, 
            self.scaler
        )
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_auc = checkpoint.get('best_auc', 0.0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_aucs = checkpoint.get('val_aucs', [])
        
        print(f"从epoch {self.start_epoch}恢复训练")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        axes[0].plot(self.train_losses, label='Train Loss', marker='o')
        axes[0].plot(self.val_losses, label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # AUC曲线
        axes[1].plot(self.val_aucs, label='Val AUC (Macro)', marker='o', color='green')
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
        
        for epoch in range(self.start_epoch, num_epochs):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss, auc_macro, auc_micro, f1_macro, f1_micro = self.validate(epoch)
            
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
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('AUC/Val_Macro', auc_macro, epoch)
            self.writer.add_scalar('AUC/Val_Micro', auc_micro, epoch)
            self.writer.add_scalar('F1/Val_Macro', f1_macro, epoch)
            self.writer.add_scalar('F1/Val_Micro', f1_micro, epoch)
            self.writer.add_scalar('LR', current_lr, epoch)
            
            # 打印信息
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"  Val AUC (Macro): {auc_macro:.4f}, Val AUC (Micro): {auc_micro:.4f}")
            print(f"  Val F1 (Macro): {f1_macro:.4f}, Val F1 (Micro): {f1_micro:.4f}")
            print(f"  LR: {current_lr:.6f}")
            print("-" * 60)
            
            # 保存检查点
            is_best = auc_macro > self.best_auc
            if is_best:
                self.best_auc = auc_macro
            
            # 保存检查点时，如果使用EMA，保存EMA模型
            if is_best and self.ema is not None:
                self.ema.apply_shadow()
                self.save_checkpoint(epoch, is_best=is_best)
                self.ema.restore()
            else:
                self.save_checkpoint(epoch, is_best=is_best)
        
        # 绘制训练曲线
        self.plot_training_curves()
        self.writer.close()
        print(f"训练完成! 最佳AUC: {self.best_auc:.4f}")


def main():
    """主函数"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = load_config(config_path)
    
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()

