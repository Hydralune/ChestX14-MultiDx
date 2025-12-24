# ConvNeXt Large 训练模块

基于ConvNeXt Large的胸部X光多标签分类训练模块，使用SOTA级别的训练方法。

## 特性

- **ConvNeXt Large Backbone**: 使用约200M参数的ConvNeXt Large作为特征提取器
- **14头分类器**: 每个头专门捕捉一类疾病，使用解耦分类头设计
- **SOTA架构组件**:
  - Spatial Attention: 聚焦病灶区域
  - Label Correlation Module: 显式建模标签相关性
  - Decoupled Class Heads: 独立的分类头，使用LayerNorm
- **SOTA训练方法**:
  - Model EMA (Exponential Moving Average): 模型集成技术
  - Asymmetric Loss (ASL): 多标签分类的SOTA损失函数
  - Mixed Precision Training: FP16混合精度训练
  - Gradient Clipping: 防止梯度爆炸
  - Early Stopping: 早停机制
  - Cosine Annealing with Warmup: 学习率调度
- **多GPU支持**: 自动检测并使用所有可用GPU（DataParallel）

## 文件结构

```
convnext/
├── model.py          # ConvNeXt Large模型定义
├── dataset.py        # 数据集加载（images和filtered_labels.csv）
├── loss.py           # 损失函数（ASL、Focal Loss等）
├── train.py          # 训练脚本（支持多GPU）
├── config.yaml       # 配置文件
└── README.md         # 本文件
```

## 环境要求

- Python 3.7+
- PyTorch 1.8+ (支持CUDA)
- timm (用于加载ConvNeXt模型)
- 其他依赖见项目根目录的requirements.txt

## 使用方法

### 1. 安装依赖

```bash
# 在项目根目录
pip install -r requirements.txt
```

### 2. 配置训练参数

编辑 `config.yaml` 文件，主要配置项：

- `batch_size`: 总batch size（4卡4090建议128，每个GPU约32）
- `learning_rate`: 学习率（ConvNeXt Large建议0.0001）
- `num_epochs`: 训练轮数（配合早停机制使用）
- `loss_type`: 损失函数类型（推荐"asymmetric_loss"）

### 3. 开始训练

```bash
# 在convnext文件夹中
cd convnext
python train.py
```

训练脚本会自动：
- 检测可用GPU数量
- 加载数据集（从项目根目录的images和filtered_labels.csv）
- 创建模型（ConvNeXt Large + 14头分类器）
- 使用SOTA训练方法进行训练
- 保存检查点和最佳模型

### 4. 恢复训练

在 `config.yaml` 中设置：

```yaml
train:
  resume: true
  resume_checkpoint: "checkpoints/checkpoint_latest.pth"
```

## 数据集路径

- **图像目录**: `../images` (相对于convnext文件夹)
- **标签文件**: `../filtered_labels.csv` (相对于convnext文件夹)

代码会自动处理路径，确保从项目根目录正确加载数据。

## 输出文件

训练过程中会生成：

- `checkpoints/checkpoint_latest.pth`: 最新检查点
- `checkpoints/checkpoint_best.pth`: 最佳模型（基于验证集AUC）
- `checkpoints/checkpoint_best_ema.pth`: 最佳EMA模型（推荐使用）
- `logs/`: TensorBoard日志文件
- `results/training_curves.png`: 训练曲线图

## 模型架构说明

### ConvNeXt Large Backbone
- 参数量: ~200M
- 特征维度: 1536
- 预训练: ImageNet

### 14头分类器设计
每个头独立预测一个疾病类别：
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

### SOTA组件

1. **Spatial Attention**: 在全局池化前应用空间注意力，聚焦病灶区域
2. **Label Correlation Module**: 
   - 使用Label Embedding学习标签表示
   - Cross-Attention机制让每个标签关注相关的视觉特征
   - 输出label-aware features
3. **Decoupled Class Heads**:
   - 每个头使用LayerNorm（而非BatchNorm）
   - 独立的Dropout和激活函数
   - 专门为单类预测优化

## 训练技巧

1. **Batch Size**: 4卡4090建议使用batch_size=128（每个GPU约32）
2. **Learning Rate**: ConvNeXt Large建议从0.0001开始
3. **Mixed Precision**: 启用FP16混合精度可以加速训练并节省显存
4. **EMA**: Model EMA通常能带来0.5-1%的AUC提升
5. **Early Stopping**: 设置patience=10，避免过拟合

## 性能优化

- **多GPU训练**: 自动使用所有可用GPU（DataParallel）
- **数据加载**: 使用16个workers加速数据加载
- **混合精度**: FP16训练，速度提升约1.5-2倍
- **Pin Memory**: 启用pin_memory加速GPU数据传输

## 注意事项

1. 确保数据集路径正确（images和filtered_labels.csv在项目根目录）
2. 4卡4090建议batch_size=128，如果显存不足可以适当减小
3. 训练过程中会保存多个检查点，推荐使用EMA版本的最佳模型
4. TensorBoard日志保存在logs文件夹，可以用 `tensorboard --logdir logs` 查看

## 参考

- ConvNeXt论文: "A ConvNet for the 2020s"
- ASL损失函数: "Asymmetric Loss For Multi-Label Classification"
- Model EMA: "Mean teachers are better role models"

