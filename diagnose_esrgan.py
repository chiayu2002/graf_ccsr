#!/usr/bin/env python3
"""
ESRGAN 预训练模型诊断工具
检查预训练模型的结构和权重键名
"""

import torch
import sys
import os

def diagnose_pretrained_model(pretrained_path):
    """诊断预训练模型"""

    print("="*60)
    print("ESRGAN 预训练模型诊断工具")
    print("="*60)
    print()

    # 1. 检查文件是否存在
    print(f"1. 检查文件: {pretrained_path}")
    if not os.path.exists(pretrained_path):
        print(f"   ❌ 文件不存在！")
        print(f"   请运行: bash download_esrgan.sh")
        return False

    file_size = os.path.getsize(pretrained_path) / (1024*1024)  # MB
    print(f"   ✓ 文件存在")
    print(f"   文件大小: {file_size:.2f} MB")
    print()

    # 2. 加载模型
    print("2. 加载模型权重...")
    try:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        print(f"   ✓ 加载成功")
    except Exception as e:
        print(f"   ❌ 加载失败: {e}")
        return False
    print()

    # 3. 检查权重格式
    print("3. 检查权重格式...")
    print(f"   checkpoint 类型: {type(checkpoint)}")

    if isinstance(checkpoint, dict):
        print(f"   字典键名: {list(checkpoint.keys())}")

        # 确定实际的权重字典
        if 'params' in checkpoint:
            state_dict = checkpoint['params']
            print(f"   ✓ 使用 'params' 键")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"   ✓ 使用 'state_dict' 键")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"   ✓ 使用 'model_state_dict' 键")
        else:
            state_dict = checkpoint
            print(f"   ✓ 直接使用 checkpoint")
    else:
        state_dict = checkpoint
        print(f"   ✓ checkpoint 就是 state_dict")
    print()

    # 4. 分析权重键名
    print("4. 分析权重键名...")
    all_keys = list(state_dict.keys())
    print(f"   总键名数量: {len(all_keys)}")
    print()

    # 找出 RRDB trunk 相关的键
    rrdb_keys = [k for k in all_keys if 'RRDB_trunk' in k or 'rrdb_trunk' in k]
    trunk_conv_keys = [k for k in all_keys if 'trunk_conv' in k]

    print(f"   RRDB trunk 相关键: {len(rrdb_keys)} 个")
    if rrdb_keys:
        print(f"   示例键名:")
        for key in rrdb_keys[:5]:
            print(f"      - {key}")
        if len(rrdb_keys) > 5:
            print(f"      ... 还有 {len(rrdb_keys)-5} 个")
    print()

    print(f"   trunk_conv 相关键: {len(trunk_conv_keys)} 个")
    if trunk_conv_keys:
        print(f"   示例键名:")
        for key in trunk_conv_keys:
            print(f"      - {key}")
    print()

    # 5. 检查第一层
    print("5. 检查第一层权重...")
    first_keys = [k for k in all_keys if k.startswith('conv_first') or k.startswith('model.conv_first')]
    if first_keys:
        print(f"   ✓ 找到 conv_first: {first_keys}")
        for key in first_keys:
            shape = state_dict[key].shape
            print(f"      {key}: {shape}")
    else:
        print(f"   ⚠️  未找到 conv_first")
    print()

    # 6. 所有唯一的前缀
    print("6. 分析键名前缀...")
    prefixes = set()
    for key in all_keys:
        parts = key.split('.')
        if len(parts) > 1:
            prefixes.add(parts[0])

    print(f"   唯一前缀: {sorted(prefixes)}")
    print()

    # 7. 显示所有键名（如果不太多）
    if len(all_keys) <= 50:
        print("7. 所有权重键名:")
        for i, key in enumerate(all_keys, 1):
            shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
            print(f"   {i:3d}. {key:60s} {shape}")
    else:
        print(f"7. 权重键名太多 ({len(all_keys)}), 只显示前20个:")
        for i, key in enumerate(all_keys[:20], 1):
            shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
            print(f"   {i:3d}. {key:60s} {shape}")
        print(f"   ... 还有 {len(all_keys)-20} 个")
    print()

    # 8. 建议
    print("="*60)
    print("诊断结果和建议:")
    print("="*60)

    if rrdb_keys:
        print("✓ 预训练模型包含 RRDB 权重")

        # 检查键名格式
        has_model_prefix = any('model.' in k for k in rrdb_keys)
        if has_model_prefix:
            print("⚠️  键名包含 'model.' 前缀")
            print("   需要在加载时移除 'model.' 前缀")
        else:
            print("✓ 键名格式正常")
    else:
        print("❌ 预训练模型不包含 RRDB 权重")
        print("   请确认下载的是正确的 RRDB_ESRGAN_x4.pth 文件")

    print()
    return True


if __name__ == "__main__":
    # 默认路径
    default_path = "pretrained_models/RRDB_ESRGAN_x4.pth"

    if len(sys.argv) > 1:
        pretrained_path = sys.argv[1]
    else:
        pretrained_path = default_path

    diagnose_pretrained_model(pretrained_path)
