"""
添加详细计时的训练脚本
用于诊断性能瓶颈
"""
import torch
import time
import sys

# 在 train.py 的主循环中添加详细计时
# 在每个关键步骤后添加 torch.cuda.synchronize() 和计时

timing_stats = {
    'data_loading': [],
    'generator_forward': [],
    'discriminator_forward': [],
    'discriminator_backward': [],
    'generator_backward': [],
    'occ_grid_update': [],
    'total_iteration': []
}

def reset_timing():
    for k in timing_stats:
        timing_stats[k] = []

def print_timing_stats():
    import numpy as np
    print("\n" + "="*70)
    print("性能分析（最近 100 步平均）")
    print("="*70)

    total_time = 0
    for name, times in timing_stats.items():
        if times:
            avg_time = np.mean(times[-100:])
            total_time += avg_time if name != 'total_iteration' else 0
            print(f"{name:25s}: {avg_time:7.2f} ms", end='')
            if name != 'total_iteration':
                pct = (avg_time / np.mean(timing_stats['total_iteration'][-100:])) * 100
                print(f"  ({pct:5.1f}%)")
            else:
                print()

    iter_time = np.mean(timing_stats['total_iteration'][-100:])
    print(f"\n每次迭代总时间: {iter_time:.2f} ms")
    print(f"吞吐量: {1000/iter_time:.1f} iterations/sec = {60000/iter_time:.1f} iterations/min")
    print("="*70 + "\n")

# 在 train.py 中的使用方法：
"""
在训练循环的开始：
    iter_start = time.time()

在数据加载后：
    torch.cuda.synchronize()
    data_time = (time.time() - data_start) * 1000
    timing_stats['data_loading'].append(data_time)

在 generator forward 后：
    torch.cuda.synchronize()
    gen_time = (time.time() - gen_start) * 1000
    timing_stats['generator_forward'].append(gen_time)

...以此类推

在迭代结束：
    torch.cuda.synchronize()
    iter_time = (time.time() - iter_start) * 1000
    timing_stats['total_iteration'].append(iter_time)

    if it % 100 == 0:
        print_timing_stats()
"""
