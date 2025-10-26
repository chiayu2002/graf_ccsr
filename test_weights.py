#!/usr/bin/env python3
"""快速测试预训练模型的键名"""
import torch

pretrained_path = 'pretrained_models/RRDB_ESRGAN_x4.pth'

try:
    checkpoint = torch.load(pretrained_path, map_location='cpu')

    # 获取实际的 state_dict
    if isinstance(checkpoint, dict):
        if 'params' in checkpoint:
            state_dict = checkpoint['params']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # 显示前10个键名
    print("前10个权重键名:")
    for i, k in enumerate(list(state_dict.keys())[:10], 1):
        print(f"{i}. {k}")

    # 统计 RRDB 相关键
    rrdb_keys = [k for k in state_dict.keys() if 'RRDB_trunk' in k or 'rrdb_trunk' in k]
    trunk_keys = [k for k in state_dict.keys() if 'trunk_conv' in k]

    print(f"\nRRDB trunk 键: {len(rrdb_keys)} 个")
    print(f"trunk_conv 键: {len(trunk_keys)} 个")

    if rrdb_keys:
        print(f"\n第一个 RRDB 键: {rrdb_keys[0]}")
    if trunk_keys:
        print(f"第一个 trunk_conv 键: {trunk_keys[0]}")

except FileNotFoundError:
    print(f"❌ 文件不存在: {pretrained_path}")
    print("请先运行: bash download_esrgan.sh")
except Exception as e:
    print(f"❌ 错误: {e}")
