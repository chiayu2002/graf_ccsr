#!/usr/bin/env python3
"""检查 trunk_conv 的键名"""
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

    # 查找包含 '2.' 的键（可能是 trunk_conv）
    keys_with_2 = [k for k in state_dict.keys() if '.2.' in k or k.startswith('model.2.')]

    print("包含 '2.' 的键名:")
    print("="*80)
    if keys_with_2:
        for k in keys_with_2[:10]:
            print(f"  {k}")
        if len(keys_with_2) > 10:
            print(f"  ... 还有 {len(keys_with_2)-10} 个")
    else:
        print("  未找到包含 '.2.' 的键")

    print()
    print("可能的 trunk_conv 候选:")
    print("="*80)

    # 查找可能的 trunk_conv
    candidates = []
    for k in state_dict.keys():
        if 'model.2' in k or 'trunk' in k.lower() or 'conv' in k.lower():
            if 'sub' not in k and 'RDB' not in k:  # 排除 RRDB 内部的
                candidates.append(k)

    if candidates:
        for k in candidates[:20]:
            print(f"  {k}")
        if len(candidates) > 20:
            print(f"  ... 还有 {len(candidates)-20} 个")
    else:
        print("  未找到明显的 trunk_conv 候选")

except FileNotFoundError:
    print(f"❌ 文件不存在: {pretrained_path}")
except Exception as e:
    print(f"❌ 错误: {e}")
