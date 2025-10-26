#!/usr/bin/env python3
"""测试 RRDB 权重键名映射"""
import re

def map_key(old_key):
    """映射预训练模型的键名到 CCSR-ESRGAN 的键名"""
    # 处理 RRDB trunk: model.1.sub.X.RDBX.convX.0.weight
    pattern1 = r'^model\.1\.sub\.(\d+)\.(.+)\.0\.(weight|bias)$'
    match = re.match(pattern1, old_key)
    if match:
        block_id, middle, param = match.groups()
        return f'rrdb_trunk.{block_id}.{middle}.{param}'

    # 处理 trunk_conv: model.2.weight
    if old_key.startswith('model.2.'):
        return old_key.replace('model.2.', 'trunk_conv.')

    return None

# 测试样例
test_keys = [
    'model.0.weight',  # conv_first (不映射)
    'model.1.sub.0.RDB1.conv1.0.weight',  # 第0个RRDB的RDB1的conv1
    'model.1.sub.0.RDB1.conv1.0.bias',
    'model.1.sub.15.RDB3.conv5.0.weight',  # 第15个RRDB的RDB3的conv5
    'model.2.weight',  # trunk_conv
    'model.2.bias',
]

print("键名映射测试:\n" + "="*80)
for old_key in test_keys:
    new_key = map_key(old_key)
    if new_key:
        print(f"✓ {old_key:45s} → {new_key}")
    else:
        print(f"✗ {old_key:45s} → (不映射)")

print("\n" + "="*80)
print("\n如果看到上面的映射结果，说明函数工作正常！")
print("重新运行训练应该能成功加载 RRDB 权重。")
