"""
快速测试脚本 - 验证环境配置和代码基本功能
"""

import sys

def test_imports():
    """测试所有必需的导入"""
    print("测试导入...")
    try:
        import torch
        import torchvision
        import timm
        import pandas
        import numpy
        import sklearn
        import matplotlib
        import yaml
        import tqdm
        print("✓ 所有包导入成功")
        print(f"  PyTorch版本: {torch.__version__}")
        print(f"  CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA设备: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_dataset():
    """测试数据集类"""
    print("\n测试数据集...")
    try:
        from dataset import CLASS_NAMES, ChestXRayDataset
        print(f"✓ 数据集类导入成功，类别数: {len(CLASS_NAMES)}")
        print(f"  类别: {CLASS_NAMES}")
        return True
    except Exception as e:
        print(f"✗ 数据集测试失败: {e}")
        return False

def test_model():
    """测试模型"""
    print("\n测试模型...")
    try:
        import torch
        from model import create_model
        model = create_model("EfficientNet-B4", num_classes=14, pretrained=False)
        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            logits = model(x)
            probs = model.predict_proba(x)
        print(f"✓ 模型测试成功")
        print(f"  输入形状: {x.shape}")
        print(f"  输出logits形状: {logits.shape}")
        print(f"  输出概率形状: {probs.shape}")
        return True
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        return False

def test_loss():
    """测试损失函数"""
    print("\n测试损失函数...")
    try:
        import torch
        from loss import create_loss
        criterion1 = create_loss('weighted_bce', pos_weight=torch.ones(14))
        criterion2 = create_loss('focal_loss')
        logits = torch.randn(2, 14)
        targets = torch.randint(0, 2, (2, 14)).float()
        loss1 = criterion1(logits, targets)
        loss2 = criterion2(logits, targets)
        print(f"✓ 损失函数测试成功")
        print(f"  Weighted BCE Loss: {loss1.item():.4f}")
        print(f"  Focal Loss: {loss2.item():.4f}")
        return True
    except Exception as e:
        print(f"✗ 损失函数测试失败: {e}")
        return False

def test_config():
    """测试配置文件"""
    print("\n测试配置文件...")
    try:
        import yaml
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✓ 配置文件加载成功")
        print(f"  模型: {config['model']['name']}")
        print(f"  批次大小: {config['train']['batch_size']}")
        print(f"  学习率: {config['train']['learning_rate']}")
        return True
    except Exception as e:
        print(f"✗ 配置文件测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("="*60)
    print("环境配置和代码功能测试")
    print("="*60)
    
    results = []
    results.append(test_imports())
    results.append(test_dataset())
    results.append(test_model())
    results.append(test_loss())
    results.append(test_config())
    
    print("\n" + "="*60)
    if all(results):
        print("✓ 所有测试通过！环境配置正确。")
        return 0
    else:
        print("✗ 部分测试失败，请检查错误信息。")
        return 1

if __name__ == '__main__':
    sys.exit(main())

