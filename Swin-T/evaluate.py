"""
评估脚本 - Swin Transformer
输出每类AUC与ROC图、预测结果CSV
"""

import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import pandas as pd

from dataset import get_data_loaders, CLASS_NAMES
from model import create_model
from metrics import calculate_metrics, calculate_per_class_auc, calculate_roc_curves
from utils import load_config


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
        model_config = self.config['model']
        
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
            random_state=self.config.get('seed', 42),
            input_channel_mode=model_config.get('input_channel_mode', 'expand')
        )
        
        print(f"测试集批次数: {len(self.test_loader)}")
    
    def _load_model(self, checkpoint_path):
        """加载模型"""
        print(f"准备加载模型: {checkpoint_path}")
        
        # 创建模型架构
        model_config = self.config['model']
        self.model = create_model(
            num_classes=model_config['num_classes'],
            pretrained=False,
            input_channel_mode=model_config.get('input_channel_mode', 'expand')
        )
        
        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型权重
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 处理键名不匹配（去掉 'module.' 前缀）
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('module.', '')
            new_state_dict[new_k] = v
        
        try:
            self.model.load_state_dict(new_state_dict)
            print("模型权重加载成功!")
        except Exception as e:
            print(f"权重加载警告: {e}")
            # 尝试部分加载
            self.model.load_state_dict(new_state_dict, strict=False)
            print("部分权重加载成功（strict=False）")
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        
        all_probs = []
        all_labels = []
        all_names = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='评估中')
            
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                logits = self.model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                # 获取图像名称
                start_idx = len(all_names)
                batch_names = []
                for i in range(len(probs)):
                    dataset_idx = start_idx + i
                    if dataset_idx < len(self.test_loader.dataset):
                        batch_names.append(self.test_loader.dataset.df.iloc[dataset_idx]['Image Index'])
                    else:
                        batch_names.append(f"image_{dataset_idx}")
                
                all_probs.append(probs)
                all_labels.append(labels_np)
                all_names.extend(batch_names)
        
        # 合并结果
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # 计算整体指标
        print("\n" + "="*60)
        print("整体评估结果")
        print("="*60)
        
        metrics = calculate_metrics(all_labels, all_probs)
        print(f"AUC (Macro): {metrics['auc_macro']:.4f}")
        print(f"AUC (Micro): {metrics['auc_micro']:.4f}")
        print(f"F1 (Macro): {metrics['f1_macro']:.4f}")
        print(f"F1 (Micro): {metrics['f1_micro']:.4f}")
        
        # 计算每类的AUC
        print("\n" + "="*60)
        print("每类AUC结果")
        print("="*60)
        
        class_aucs = calculate_per_class_auc(all_labels, all_probs, CLASS_NAMES)
        for i, class_name in enumerate(CLASS_NAMES):
            print(f"{class_name:20s}: {class_aucs[i]:.4f}")
        
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
            'auc_macro': metrics['auc_macro'],
            'auc_micro': metrics['auc_micro'],
            'f1_macro': metrics['f1_macro'],
            'f1_micro': metrics['f1_micro'],
            'class_aucs': class_aucs
        }
    
    def plot_roc_curves(self, labels, probs, class_aucs):
        """绘制ROC曲线"""
        num_classes = len(CLASS_NAMES)
        
        # 绘制所有类的ROC曲线
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
                axes[i].set_xlabel('False Positive Rate')
                axes[i].set_ylabel('True Positive Rate')
                axes[i].set_title(f'{CLASS_NAMES[i]}')
                axes[i].legend(loc="lower right")
                axes[i].grid(True)
        
        # 隐藏多余的子图
        for i in range(num_classes, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        roc_path = os.path.join(self.config['paths']['result_dir'], 'roc_curves.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        print(f"ROC曲线已保存: {roc_path}")
        plt.close()
        
        # 绘制平均ROC曲线
        self.plot_mean_roc_curve(labels, probs, class_aucs)
    
    def plot_mean_roc_curve(self, labels, probs, class_aucs):
        """绘制平均ROC曲线"""
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
    
    args = parser.parse_args()
    
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = load_config(config_path)
    
    evaluator = Evaluator(config, args.checkpoint)
    results = evaluator.evaluate()
    
    print("\n评估完成!")


if __name__ == '__main__':
    main()

