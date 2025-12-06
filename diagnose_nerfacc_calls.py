"""
诊断 NerfAcc 渲染的实际调用次数
添加详细的计时和调用计数
"""
import torch
import time

# 在 run_nerf_mod.py 的 render_nerfacc 函数中添加计数器
RENDER_NERFACC_STATS = {
    'total_calls': 0,
    'batch_loops': 0,
    'sampling_calls': 0,
    'sigma_fn_calls': 0,
    'total_sigma_queries': 0,
    'time_in_sampling': 0,
    'time_in_rendering': 0,
}

def reset_stats():
    global RENDER_NERFACC_STATS
    for k in RENDER_NERFACC_STATS:
        RENDER_NERFACC_STATS[k] = 0

def print_stats():
    stats = RENDER_NERFACC_STATS
    print("\n" + "="*70)
    print("NerfAcc 渲染统计")
    print("="*70)
    print(f"render_nerfacc 调用次数: {stats['total_calls']}")
    print(f"batch 循环次数: {stats['batch_loops']}")
    print(f"estimator.sampling() 调用次数: {stats['sampling_calls']}")
    print(f"sigma_fn 调用次数: {stats['sigma_fn_calls']}")
    print(f"总密度查询点数: {stats['total_sigma_queries']:,}")
    print(f"\n时间统计:")
    print(f"  采样时间: {stats['time_in_sampling']:.2f}ms")
    print(f"  渲染时间: {stats['time_in_rendering']:.2f}ms")
    print(f"  总时间: {stats['time_in_sampling'] + stats['time_in_rendering']:.2f}ms")

    if stats['sampling_calls'] > 0:
        avg_sigma_per_sampling = stats['total_sigma_queries'] / stats['sampling_calls']
        print(f"\n平均每次 sampling 查询: {avg_sigma_per_sampling:.0f} 个点")

    if stats['total_calls'] > 0:
        avg_batch_loops = stats['batch_loops'] / stats['total_calls']
        avg_sampling_per_call = stats['sampling_calls'] / stats['total_calls']
        print(f"\n平均每次 render_nerfacc:")
        print(f"  batch 循环: {avg_batch_loops:.0f} 次")
        print(f"  sampling 调用: {avg_sampling_per_call:.0f} 次")

    print("="*70 + "\n")

# 在 run_nerf_mod.py 中插入计数的位置：
"""
在 render_nerfacc 开始处（line 120）:
    RENDER_NERFACC_STATS['total_calls'] += 1

在 for b in range(bs) 循环中（line 160）:
    RENDER_NERFACC_STATS['batch_loops'] += 1

在 estimator.sampling() 之前（line 250）:
    RENDER_NERFACC_STATS['sampling_calls'] += 1
    sampling_start = time.time()

在 estimator.sampling() 之后（line 261）:
    RENDER_NERFACC_STATS['time_in_sampling'] += (time.time() - sampling_start) * 1000

在 sigma_fn 内部（line 243）:
    RENDER_NERFACC_STATS['sigma_fn_calls'] += 1
    RENDER_NERFACC_STATS['total_sigma_queries'] += positions.shape[0]
"""

# 使用方法：
# 在训练脚本中:
# from run_nerf_mod import RENDER_NERFACC_STATS, reset_stats, print_stats
#
# reset_stats()
# # ... 训练 100 步 ...
# print_stats()
