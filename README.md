# 胸部X光多标签分类项目

基于深度学习的胸部X光图像多标签分类系统，使用PyTorch实现。

## 项目简介

本项目实现了对胸部X光正位图像（PA/AP）的14类疾病多标签分类任务。每张图像可能同时包含0到多种疾病标签。

### 数据集信息
- **样本数量**: 约21,844张图像
- **标签数量**: 14类疾病
- **标签格式**: 字符串，用 `|` 分隔（如 `Cardiomegaly|Effusion`）
- **No Finding**: 表示无疾病

### 14类疾病标签
1. Atelectasis（肺不张）
2. Cardiomegaly（心脏扩大）
3. Consolidation（实变）
4. Edema（水肿）
5. Effusion（积液）
6. Emphysema（肺气肿）
7. Fibrosis（纤维化）
8. Hernia（疝）
9. Infiltration（浸润）
10. Mass（肿块）
11. Nodule（结节）
12. Pleural_Thickening（胸膜增厚）
13. Pneumonia（肺炎）
14. Pneumothorax（气胸）

## 技术特性

- **模型架构**: EfficientNet-B4 或 DenseNet121
- **输入尺寸**: 512×512
- **损失函数**: Weighted BCE 或 Focal Loss（处理类别不平衡）
- **评估指标**: AUC (macro/micro) + F1 (macro/micro)
- **训练特性**:
  - 混合精度训练（Mixed Precision）
  - Resume训练支持
  - 自动保存最优模型
  - TensorBoard可视化
- **评估特性**:
  - 每类AUC计算
  - ROC曲线绘制
  - 训练曲线可视化

## 环境配置

### 系统要求
- Python 3.7+
- CUDA支持的GPU（推荐）

### 安装依赖

```bash
pip install -r requirements.txt
```

## 项目结构

```
ChestX14-MultiDx/
├── config.yaml          # 配置文件
├── dataset.py           # 数据集类和数据加载
├── model.py             # 模型定义
├── loss.py              # 损失函数
├── train.py             # 训练脚本
├── evaluate.py          # 评估脚本
├── requirements.txt     # 依赖包
├── README.md           # 说明文档
├── filtered_labels.csv  # 标签CSV文件
├── images/             # 图像目录
├── checkpoints/        # 模型检查点（训练后生成）
├── logs/               # TensorBoard日志（训练后生成）
└── results/            # 评估结果（评估后生成）
```

## 配置文件说明

主要配置在 `config.yaml` 中：

- **data**: 数据路径和划分比例
- **model**: 模型名称和参数
- **train**: 训练超参数（学习率、批次大小、损失函数等）
- **image**: 图像预处理参数
- **paths**: 输出路径

## 使用方法

### 1. 准备数据

确保数据文件结构如下：
```
ChestX14-MultiDx/
├── filtered_labels.csv    # 包含列: Image Index, Finding Labels, Patient ID, ...
└── images/                # 图像文件夹
    ├── 00000001_000.png
    ├── 00000001_001.png
    └── ...
```

### 2. 配置参数

编辑 `config.yaml` 调整训练参数：

```yaml
model:
  name: "EfficientNet-B4"  # 或 "DenseNet121"

train:
  batch_size: 16
  num_epochs: 50
  learning_rate: 0.0001
  loss_type: "weighted_bce"  # 或 "focal_loss"
  mixed_precision: true      # 启用混合精度训练
```

### 3. 开始训练

```bash
python train.py
```

训练过程中会：
- 自动保存最新检查点到 `checkpoints/checkpoint_latest.pth`
- 保存最佳模型到 `checkpoints/checkpoint_best.pth`
- 生成TensorBoard日志到 `logs/`
- 训练结束后生成训练曲线到 `results/training_curves.png`

### 4. 恢复训练

如果需要从检查点恢复训练，修改 `config.yaml`：

```yaml
train:
  resume: true
  resume_checkpoint: "checkpoints/checkpoint_latest.pth"
```

然后运行训练命令即可。

### 5. 评估模型

```bash
python evaluate.py --checkpoint checkpoints/checkpoint_best.pth
```

评估结果将保存在 `results/` 目录：
- `class_aucs.csv`: 每类AUC结果
- `roc_curves.png`: 每类ROC曲线
- `mean_roc_curve.png`: 平均ROC曲线
- `predictions.csv`: 详细预测结果

### 6. 查看训练过程

使用TensorBoard查看训练过程：

```bash
tensorboard --logdir logs
```

然后在浏览器打开 `http://localhost:6006`

## 关键特性说明

### 患者级数据划分

为确保数据泄露，数据集按患者ID划分，同一患者的所有图像都在同一集合（训练/验证/测试）中。

### 类别不平衡处理

1. **Weighted BCE**: 自动计算正样本权重 `pos_weight`，平衡正负样本
2. **Focal Loss**: 通过聚焦参数减少易分类样本的权重，关注难样本

### 混合精度训练

使用 `torch.cuda.amp` 自动混合精度训练，可以：
- 加速训练
- 减少显存占用
- 保持训练精度

### 数据增强

训练时使用数据增强：
- 随机水平翻转
- 随机旋转（±10度）
- 颜色抖动（亮度、对比度）

## 评估指标

### AUC (Area Under Curve)
- **Macro AUC**: 每类AUC的平均值
- **Micro AUC**: 将所有样本合并计算的AUC

### F1 Score
- **Macro F1**: 每类F1的平均值
- **Micro F1**: 全局F1分数

### 每类AUC

评估脚本会输出每一类疾病的AUC值，便于分析各类疾病的分类性能。

## 模型权重

训练过程中会保存：
- **checkpoint_latest.pth**: 最新epoch的检查点
- **checkpoint_best.pth**: 验证集AUC最高的模型

检查点包含：
- 模型权重
- 优化器状态
- 学习率调度器状态
- 训练历史（损失、AUC等）

## 注意事项

1. **显存要求**: EfficientNet-B4 + batch_size=16 大约需要8GB+显存
   - 如果显存不足，可以减小 `batch_size` 或使用 `mixed_precision`
   
2. **数据划分**: 默认按 7:1.5:1.5 划分训练/验证/测试集
   - 可在 `config.yaml` 中调整比例
   - 确保比例之和为1.0

3. **随机种子**: 代码中设置了随机种子确保可复现
   - 可在 `config.yaml` 中修改 `seed`

4. **预训练模型**: 默认使用ImageNet预训练权重
   - 如果网络无法下载，可设置 `pretrained: false`

## 故障排查

### 常见问题

1. **CUDA out of memory**
   - 减小 `batch_size`
   - 启用 `mixed_precision`
   - 使用较小的模型（如DenseNet121）

2. **数据加载慢**
   - 检查 `num_workers` 设置
   - 确保图像路径正确

3. **类别AUC为N/A**
   - 某些类别在测试集中可能没有正样本
   - 这是正常现象

## 许可证

本项目仅供学习使用。

## 联系方式

如有问题，请查看代码注释或提交Issue。

