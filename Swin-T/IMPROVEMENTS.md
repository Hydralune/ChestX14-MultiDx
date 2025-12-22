# Swin-T 模型性能提升方案

## 当前状态
- **最佳 AUC**: 0.7841
- **模型**: Swin-T (Tiny)
- **图像尺寸**: 512x512 → **640x640** (已提升)
- **损失函数**: Asymmetric Loss

## 重要说明：Mixup 不适合医学图像

**Mixup 已默认关闭**，原因：
1. **破坏解剖结构**：医学图像需要精确的空间关系，Mixup会混合两个样本，破坏解剖结构
2. **模糊小病灶**：病灶通常很小，Mixup可能导致重要特征被模糊
3. **标签混淆**：多标签分类中，Mixup可能导致不合理的标签组合
4. **实际验证**：关闭Mixup后性能有提升

## 已实现的改进

### 1. EMA (Exponential Moving Average) ✅
- **作用**: 对模型权重进行指数移动平均，提升模型稳定性和性能
- **配置**: `use_ema: true`, `ema_decay: 0.9999`
- **预期提升**: +0.5-1% AUC

### 2. 改进的学习率调度器 ✅
- **改进**: Cosine Annealing with Warmup
- **配置**: `warmup_epochs: 5`, `eta_min: 1e-6`
- **优势**: 更平滑的学习率变化，避免训练初期的不稳定
- **预期提升**: +0.3-0.5% AUC

### 3. Asymmetric Loss ✅
- **作用**: 专门为多标签分类设计的损失函数，对正负样本使用不同的gamma
- **配置**: `loss_type: "asymmetric_loss"`, `asymmetric_gamma_neg: 4`, `asymmetric_gamma_pos: 1`
- **优势**: 更好地处理类别不平衡和难样本
- **预期提升**: +1-2% AUC

### 4. Mixup 数据增强 ❌
- **状态**: **已关闭**（医学图像不适合）
- **原因**: 会破坏解剖结构，模糊小病灶，导致标签混淆
- **建议**: 保持关闭状态

### 5. 梯度累积 ✅
- **作用**: 模拟更大的batch size，提升训练稳定性
- **配置**: `gradient_accumulation_steps: 1` (可根据显存调整)
- **优势**: 允许使用更大的有效batch size而不增加显存

### 6. 修复 autocast 警告 ✅
- **改进**: 使用新的 `torch.amp.autocast('cuda')` API
- **优势**: 消除 FutureWarning，代码更规范

## 配置文件更新

主要更新了 `config.yaml`：

```yaml
model:
  model_size: "tiny"  # 可选: "tiny", "small", "base"
  
image:
  size: 640  # 提升到640 (可尝试768/896)

train:
  batch_size: 12  # 640分辨率建议使用12
  num_epochs: 150  # 增加训练轮数
  
  # 损失函数配置
  loss_type: "asymmetric_loss"  # 推荐使用
  asymmetric_gamma_neg: 4
  asymmetric_gamma_pos: 1
  asymmetric_clip: 0.05
  
  # EMA配置
  use_ema: true
  ema_decay: 0.9999
  
  # Mixup配置（已关闭）
  use_mixup: false  # 医学图像不适合
  
  # 学习率调度
  warmup_epochs: 5
  eta_min: 1e-6
```

### 7. 更高精度方案 ✅
- **更大图像尺寸**: 512 → 640 (可尝试768/896)
- **更大模型**: 支持 Swin-S (50M) 和 Swin-B (88M)
- **更长训练**: 100 → 150 epochs
- **预期提升**: +1-2% AUC

## 预期性能提升

综合以上改进（不包括Mixup），预期 AUC 可以从 **0.7841** 提升到 **0.80-0.82** (+1.6-3.6%)。

**关键提升点**：
1. Asymmetric Loss: +1-2%
2. EMA: +0.5-1%
3. 更大图像尺寸 (640): +0.5-1%
4. 更大模型 (Swin-S): +0.5-1%

## 使用建议

### 1. 损失函数选择
- **Asymmetric Loss**: 推荐用于多标签分类，通常效果最好 ✅
- **Combined Loss**: 如果想结合 Weighted BCE 和 Asymmetric Loss 的优势
- **Weighted BCE**: 如果 Asymmetric Loss 效果不理想，可以回退

### 2. 模型大小选择
- **Swin-T (tiny)**: 28M参数，显存需求低，速度快
- **Swin-S (small)**: 50M参数，**推荐**，平衡性能和速度
- **Swin-B (base)**: 88M参数，最高精度，需要更多显存

### 3. 图像尺寸选择
- **512**: 基础配置，显存需求低
- **640**: **推荐**，平衡精度和显存
- **768/896**: 最高精度，需要更多显存（可能需要减小batch_size）

### 4. EMA 参数调整
- `ema_decay` 越大，EMA 更新越慢，模型越稳定（推荐 0.999-0.9999）
- 如果显存充足，可以增加 `ema_decay`

### 5. 梯度累积
- 如果显存不足，可以增加 `gradient_accumulation_steps`（如 2-4）
- 有效 batch size = `batch_size * gradient_accumulation_steps`

### 6. 显存优化
如果使用更大模型或图像尺寸导致显存不足：
- 减小 `batch_size`（如 8-12）
- 增加 `gradient_accumulation_steps`（如 2-4）
- 使用 `mixed_precision: true`（已默认开启）

## 进一步优化方向

如果以上改进后仍想继续提升，可以考虑：

1. **更大的模型**: ✅ 已支持 Swin-S 和 Swin-B
2. **更大的图像尺寸**: ✅ 已提升到 640，可尝试 768/896
3. **更长的训练**: ✅ 已增加到 150 epochs
4. **Test-Time Augmentation (TTA)**: ✅ 已实现（在evaluate.py中使用）
5. **模型集成**: 训练多个模型（不同随机种子）并集成
6. **伪标签**: 使用高置信度的预测作为额外训练数据
7. **多尺度训练**: 训练时随机使用不同尺寸（需要修改dataset）
8. **标签平滑**: 对多标签分类可能有效（需要实验验证）

## 推荐配置（高精度）

```yaml
model:
  model_size: "small"  # 或 "base"

image:
  size: 640  # 或 768

train:
  batch_size: 8  # 根据显存调整
  num_epochs: 150
  loss_type: "asymmetric_loss"
  use_ema: true
  use_mixup: false  # 保持关闭
```

## 训练命令

```bash
cd Swin-T
python train.py
```

训练会自动使用 `config.yaml` 中的配置。

