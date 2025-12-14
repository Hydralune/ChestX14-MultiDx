"""
评估脚本 (High Performance Version)
支持 EMA 权重加载 + TTA (Test Time Augmentation)
"""

import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, auc
from tqdm import tqdm
import pandas as pd

from dataset import get_data_loaders, CLASS_NAMES
from model import create_model


class Evaluator:
    """评估器类"""
    
    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建结果目录
        os.makedirs(config['paths']['result_dir'], exist_ok=True)
        
        # 初始化数据加载器
        self._init_dataloaders()
        
        # 加载模型
        self._load_model(checkpoint_path)
    
    def _init_dataloaders(self):
        """初始化数据加载器"""
        data_config = self.config['data']
        image_config = self.config['image']
        
        _, _, self.test_loader = get_data_loaders(
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
        
        print(f"测试集批次数: {len(self.test_loader)}")
    
    def _load_model(self, checkpoint_path):
        """加载模型 (优先尝试加载 EMA 权重)"""
        print(f"准备加载模型: {checkpoint_path}")
        
        # 创建模型架构
        model_config = self.config['model']
        self.model = create_model(
            model_name=model_config['name'],
            num_classes=model_config['num_classes'],
            pretrained=False
        )
        
        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # === 关键修改：智能识别 EMA 权重 ===
        if 'model_state_dict' in checkpoint:
            # 这是一个标准的 checkpoint 字典
            state_dict = checkpoint['model_state_dict']
            print("检测到标准 Checkpoint")
            
            # 如果存在 EMA 状态且不在 best_ema 文件中，尝试加载 EMA
            if 'model_ema_state_dict' in checkpoint:
                print(">> 注意: 发现 EMA 权重，正在加载 EMA 权重以获得更好性能...")
                state_dict = checkpoint['model_ema_state_dict']
            
        else:
            # 这可能是直接保存的 state_dict 或者 best_ema 文件
            state_dict = checkpoint
            print("检测到纯权重文件 (或 EMA 文件)")

        # 加载参数
        try:
            self.model.load_state_dict(state_dict)
            print("模型权重加载成功!")
        except Exception as e:
            print(f"权重加载警告: {e}")
            print("尝试处理键名不匹配 (去掉 'module.' 前缀)...")
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict)
            print("修复后加载成功!")

        self.model = self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self, use_tta=True):
        """
        评估模型
        Args:
            use_tta: 是否使用测试时增强 (Test Time Augmentation)
        """
        print(f"开始评估... (TTA开启: {use_tta})")
        
        all_probs = []
        all_labels = []
        all_names = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='评估中')
            
            for images, labels, names in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # === TTA (Test Time Augmentation) 核心逻辑 ===
                if use_tta:
                    # 1. 原图预测
                    logits_orig = self.model(images)
                    probs_orig = torch.sigmoid(logits_orig)
                    
                    # 2. 水平翻转预测
                    images_flip = torch.flip(images, dims=[3])
                    logits_flip = self.model(images_flip)
                    probs_flip = torch.sigmoid(logits_flip)
                    
                    # 3. 平均
                    probs = (probs_orig + probs_flip) / 2.0
                else:
                    logits = self.model(images)
                    probs = torch.sigmoid(logits)
                # ==========================================
                
                probs = probs.cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                all_probs.append(probs)
                all_labels.append(labels_np)
                all_names.extend(names)
        
        # 合并结果
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # 计算整体指标
        print("\n" + "="*60)
        print("整体评估结果")
        print("="*60)
        
        # AUC
        try:
            auc_macro = roc_auc_score(all_labels, all_probs, average='macro')
            auc_micro = roc_auc_score(all_labels, all_probs, average='micro')
            print(f"AUC (Macro): {auc_macro:.4f}")
            print(f"AUC (Micro): {auc_micro:.4f}")
        except Exception as e:
            print(f"AUC计算错误: {e}")
            auc_macro = 0.0
            auc_micro = 0.0
        
        # F1
        preds = (all_probs > 0.5).astype(int)
        f1_macro = f1_score(all_labels, preds, average='macro', zero_division=0)
        f1_micro = f1_score(all_labels, preds, average='micro', zero_division=0)
        print(f"F1 (Macro): {f1_macro:.4f}")
        print(f"F1 (Micro): {f1_micro:.4f}")
        
        # 计算每类的AUC
        print("\n" + "="*60)
        print("每类AUC结果")
        print("="*60)
        
        class_aucs = []
        for i, class_name in enumerate(CLASS_NAMES):
            try:
                class_auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                class_aucs.append(class_auc)
                print(f"{class_name:20s}: {class_auc:.4f}")
            except:
                class_aucs.append(0.0)
                print(f"{class_name:20s}: N/A (无正样本)")
        
        # 保存结果到CSV
        results_df = pd.DataFrame({
            'Class': CLASS_NAMES,
            'AUC': class_aucs
        })
        results_path = os.path.join(self.config['paths']['result_dir'], 'class_aucs.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\n每类AUC结果已保存: {results_path}")
        
        # 绘制ROC曲线
        self.plot_roc_curves(all_labels, all_probs, class_aucs)
        
        # 保存预测结果
        self.save_predictions(all_probs, all_labels, all_names)
        
        return {
            'auc_macro': auc_macro,
            'auc_micro': auc_micro,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'class_aucs': class_aucs
        }
    
    def plot_roc_curves(self, labels, probs, class_aucs):
        """绘制ROC曲线"""
        num_classes = len(CLASS_NAMES)
        
        # 计算每类的ROC曲线
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        axes = axes.flatten()
        
        for i in range(num_classes):
            if i < len(axes):
                fpr, tpr, _ = roc_curve(labels[:, i], probs[:, i])
                roc_auc = class_aucs[i]
                
                axes[i].plot(fpr, tpr, color='darkorange', lw=2, 
                           label=f'AUC = {roc_auc:.3f}')
                axes[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                axes[i].set_xlim([0.0, 1.0])
                axes[i].set_ylim([0.0, 1.05])
                axes[i].set_xlabel('FPR')
                axes[i].set_ylabel('TPR')
                axes[i].set_title(f'{CLASS_NAMES[i]}')
                axes[i].legend(loc="lower right")
                axes[i].grid(True)
        
        # 隐藏多余的子图
        for i in range(num_classes, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        roc_path = os.path.join(self.config['paths']['result_dir'], 'roc_curves.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        print(f"ROC曲线已保存: {roc_path}")
        plt.close()
        
        # 绘制平均ROC曲线
        self.plot_mean_roc_curve(labels, probs, class_aucs)
    
    def plot_mean_roc_curve(self, labels, probs, class_aucs):
        """绘制平均ROC曲线"""
        # 计算macro-average ROC
        all_fpr = np.unique(np.concatenate([
            roc_curve(labels[:, i], probs[:, i])[0] 
            for i in range(len(CLASS_NAMES))
        ]))
        
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(CLASS_NAMES)):
            fpr, tpr, _ = roc_curve(labels[:, i], probs[:, i])
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        
        mean_tpr /= len(CLASS_NAMES)
        mean_auc = np.mean(class_aucs)
        
        plt.figure(figsize=(8, 8))
        plt.plot(all_fpr, mean_tpr, color='darkorange', lw=2,
                label=f'Macro-average ROC (AUC = {mean_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Macro-average ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        mean_roc_path = os.path.join(self.config['paths']['result_dir'], 'mean_roc_curve.png')
        plt.savefig(mean_roc_path, dpi=300, bbox_inches='tight')
        print(f"平均ROC曲线已保存: {mean_roc_path}")
        plt.close()
    
    def save_predictions(self, probs, labels, names):
        """保存预测结果"""
        results = {
            'Image': names,
        }
        
        # 添加每类的概率和预测
        for i, class_name in enumerate(CLASS_NAMES):
            results[f'{class_name}_prob'] = probs[:, i]
            results[f'{class_name}_pred'] = (probs[:, i] > 0.5).astype(int)
            results[f'{class_name}_true'] = labels[:, i]
        
        results_df = pd.DataFrame(results)
        results_path = os.path.join(self.config['paths']['result_dir'], 'predictions.csv')
        results_df.to_csv(results_path, index=False)
        print(f"预测结果已保存: {results_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='评估模型')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    # 默认开启 TTA
    parser.add_argument('--no-tta', action='store_true', help='关闭 TTA')
    
    args = parser.parse_args()
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建评估器
    evaluator = Evaluator(config, args.checkpoint)
    
    # 评估 (使用 TTA)
    results = evaluator.evaluate(use_tta=not args.no_tta)
    
    print("\n评估完成!")


if __name__ == '__main__':
    main()