# Swin Transformer 胸部X光14类多标签分类

基于 Swin Transformer 的胸部X光多标签分类子项目，使用 timm 的 `swin_tiny_patch4_window7_224` 作为 backbone。

## 项目结构

```
Swin-T/
├── config.yaml          # 配置文件
├── labels.py            # 14类标签定义与编码/解码
├── dataset.py           # 数据集加载（患者级划分）
├── model.py             # Swin-T 模型定义
├── loss.py              # 损失函数（Weighted BCE / Focal Loss）
├── metrics.py           # 评估指标计算
├── train.py             # 训练脚本
├── evaluate.py          # 评估脚本
├── utils.py             # 工具函数
├── requirements.txt     # 依赖包
├── README.md            # 说明文档
├── checkpoints/         # 模型检查点
├── logs/                # TensorBoard日志
└── results/             # 评估结果
```

## 环境配置

### 安装依赖

```bash
cd Swin-T
pip install -r requirements.txt
```

## 数据准备

数据路径相对于仓库根目录：
- CSV文件：`../filtered_labels.csv`
- 图像目录：`../images/`

确保这些文件存在。

## 配置说明

主要配置项在 `config.yaml` 中：

### 模型配置

- `model.name`: 模型名称（固定为 "Swin-T"）
- `model.pretrained`: 是否使用ImageNet预训练权重
- `model.input_channel_mode`: 输入通道处理方式
  - `"expand"`: 将1通道灰度图扩展到3通道（复制3次）- **推荐**
  - `"modify"`: 修改第一层适配1通道输入

### 训练配置

- `train.batch_size`: 批次大小（默认16）
- `train.num_epochs`: 训练轮数（默认100）
- `train.learning_rate`: 学习率（默认0.0001）
- `train.loss_type`: 损失函数类型（`weighted_bce` 或 `focal_loss`）
- `train.mixed_precision`: 是否使用混合精度训练（默认true）
- `train.scheduler`: 学习率调度器（`cosine` 或 `linear`）

### 数据划分

- `data.train_ratio`: 训练集比例（默认0.7）
- `data.val_ratio`: 验证集比例（默认0.15）
- `data.test_ratio`: 测试集比例（默认0.15）

**注意**：数据集按患者ID划分，确保同一患者的所有图像在同一集合中。

## 使用方法

### 1. 训练模型

```bash
cd Swin-T
python train.py
```

训练过程会：
- 自动保存 `checkpoint_latest.pth`（最新epoch）
- 自动保存 `checkpoint_best.pth`（最佳AUC模型）
- 记录TensorBoard日志到 `logs/`
- 训练结束后保存训练曲线到 `results/training_curves.png`

### 2. 恢复训练

在 `config.yaml` 中设置：

```yaml
train:
  resume: true
  resume_checkpoint: "checkpoints/checkpoint_latest.pth"
```

然后运行 `python train.py`。

### 3. 评估模型

```bash
python evaluate.py --checkpoint checkpoints/checkpoint_best.pth
```

评估结果会保存到 `results/`：
- `class_aucs.csv`: 每类AUC值
- `roc_curves.png`: 所有类的ROC曲线
- `mean_roc_curve.png`: 平均ROC曲线
- `predictions.csv`: 预测结果（概率、预测、真实标签）

### 4. 查看训练过程

```bash
tensorboard --logdir logs
```

然后在浏览器打开 `http://localhost:6006`

## 关键特性

### 1. 输入通道处理

Swin Transformer 默认接受3通道输入，而X光图是1通道灰度图。本项目支持两种处理方式：

**方式1：expand（推荐）**
- 在数据变换中将1通道复制为3通道
- 无需修改模型结构
- 配置：`model.input_channel_mode: "expand"`

**方式2：modify**
- 修改模型第一层以适配1通道输入
- 权重初始化：将3通道权重的平均值复制到1通道
- 配置：`model.input_channel_mode: "modify"`

### 2. 输入分辨率适配

Swin Transformer 默认输入224×224，本项目适配到512×512：
- 在数据变换中直接 Resize 到 512×512
- 无需修改模型结构（Swin支持任意输入尺寸）

### 3. 患者级数据划分

数据集按患者ID划分，确保：
- 同一患者的所有图像在同一集合（训练/验证/测试）
- 避免数据泄露
- 更符合实际应用场景

### 4. 类别不平衡处理

- **Weighted BCE**: 自动计算正样本权重 `pos_weight`，平衡正负样本
- **Focal Loss**: 通过聚焦参数减少易分类样本的权重，关注难样本

### 5. 混合精度训练

使用 `torch.cuda.amp` 自动混合精度训练：
- 加速训练
- 减少显存占用
- 保持训练精度

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

1. **显存要求**: Swin-T + batch_size=16 大约需要6GB+显存
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
   - 启用 `mixed_precision: true`
   - 减少 `num_workers`

2. **模型加载失败**
   - 检查checkpoint路径是否正确
   - 确认模型配置（`input_channel_mode`）与训练时一致

3. **数据路径错误**
   - 确认 `filtered_labels.csv` 和 `images/` 目录在仓库根目录
   - 检查 `config.yaml` 中的路径配置

## 参考

- [timm文档](https://github.com/rwightman/pytorch-image-models)
- [Swin Transformer论文](https://arxiv.org/abs/2103.14030)




