"""
数据验证脚本 - 检查数据是否正确加载
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from graf.config import get_data, load_config

def check_data(config_path='configs/default.yaml'):
    print("="*60)
    print("数据验证检查")
    print("="*60)

    # 加载配置
    config = load_config(config_path)
    print(f"\n配置文件: {config_path}")
    print(f"数据路径: {config['data']['datadir']}")
    print(f"图像尺寸: {config['data']['imsize']}")
    print(f"Batch size: {config['training']['batch_size']}")

    # 获取数据加载器
    try:
        train_loader = get_data(config)
        print(f"\n✓ 数据加载器创建成功")
        print(f"  数据集大小: {len(train_loader.dataset)}")
    except Exception as e:
        print(f"\n✗ 数据加载失败: {e}")
        return

    # 检查第一个batch
    try:
        x, label = next(iter(train_loader))
        print(f"\n✓ 成功加载第一个batch")
        print(f"  图像形状: {x.shape}")
        print(f"  标签形状: {label.shape}")
        print(f"  图像范围: [{x.min().item():.3f}, {x.max().item():.3f}]")
        print(f"  图像均值: {x.mean().item():.3f}")
        print(f"  图像标准差: {x.std().item():.3f}")
    except Exception as e:
        print(f"\n✗ 加载batch失败: {e}")
        return

    # 可视化检查
    print(f"\n检查图像内容...")

    # 统计
    print(f"\n图像统计:")
    print(f"  黑色像素 (< 0.1): {(x < 0.1).sum().item() / x.numel() * 100:.1f}%")
    print(f"  白色像素 (> 0.9): {(x > 0.9).sum().item() / x.numel() * 100:.1f}%")
    print(f"  中间像素: {((x >= 0.1) & (x <= 0.9)).sum().item() / x.numel() * 100:.1f}%")

    # 保存样本图像
    try:
        # 转换为numpy并调整维度
        sample_img = x[0].cpu().numpy()  # [C, H, W]
        if sample_img.shape[0] == 3:  # RGB
            sample_img = np.transpose(sample_img, (1, 2, 0))  # [H, W, C]
        elif sample_img.shape[0] == 1:  # 灰度
            sample_img = sample_img[0]  # [H, W]

        plt.figure(figsize=(8, 8))
        if len(sample_img.shape) == 3:
            plt.imshow(sample_img)
        else:
            plt.imshow(sample_img, cmap='gray')
        plt.title(f'Sample Image - Label: {label[0].cpu().numpy()}')
        plt.axis('off')
        plt.savefig('data_sample.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ 样本图像已保存到: data_sample.png")
    except Exception as e:
        print(f"\n✗ 保存图像失败: {e}")

    # 检查多个batch的一致性
    print(f"\n检查多个batch...")
    batch_stats = []
    for i, (x, label) in enumerate(train_loader):
        if i >= 5:  # 检查前5个batch
            break
        batch_stats.append({
            'mean': x.mean().item(),
            'std': x.std().item(),
            'min': x.min().item(),
            'max': x.max().item(),
        })

    print(f"\n前5个batch的统计:")
    for i, stats in enumerate(batch_stats):
        print(f"  Batch {i}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
              f"range=[{stats['min']:.3f}, {stats['max']:.3f}]")

    # 检查一致性
    means = [s['mean'] for s in batch_stats]
    stds = [s['std'] for s in batch_stats]

    if max(means) - min(means) > 0.1:
        print(f"\n⚠ 警告: batch间均值差异较大 ({max(means)-min(means):.3f})")
    else:
        print(f"\n✓ batch间均值一致")

    if max(stds) - min(stds) > 0.1:
        print(f"⚠ 警告: batch间标准差差异较大 ({max(stds)-min(stds):.3f})")
    else:
        print(f"✓ batch间标准差一致")

    # 检查标签
    print(f"\n检查标签...")
    print(f"  标签示例: {label[:3].cpu().numpy()}")
    print(f"  标签范围: [{label.min().item():.1f}, {label.max().item():.1f}]")

    print(f"\n" + "="*60)
    print("数据验证完成")
    print("="*60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()

    check_data(args.config)
