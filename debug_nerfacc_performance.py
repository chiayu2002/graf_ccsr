"""
NerfAcc 性能调试脚本
用于对比原始render和render_nerfacc的性能差异
"""

import torch
import time
import numpy as np
from graf.config import load_config, build_models
from graf.utils import count_trainable_parameters, to_device
from graf.distributions import get_zdist
import argparse

def test_render_performance(config, num_iterations=10):
    """测试渲染性能"""
    device = torch.device('cuda')

    # 初始化模型
    print("初始化模型...")
    generator_test, _ = build_models(config, disc=False)
    generator_test = generator_test.to(device)
    generator_test.eval()

    # 准备测试数据
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)
    batch_size = config['training']['batch_size']

    # 创建fake label
    label = torch.zeros(batch_size, 3, device=device)
    for i in range(batch_size):
        label[i, 2] = i * (360 / batch_size)  # 均匀分布的角度

    print(f"\n{'='*60}")
    print(f"测试配置:")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations: {num_iterations}")
    print(f"  NerfAcc enabled: {generator_test.use_nerfacc}")
    if generator_test.use_nerfacc:
        print(f"  OccGrid resolution: {generator_test.estimator.resolution}")
        print(f"  Render step size: {generator_test.render_step_size}")
    print(f"{'='*60}\n")

    # ========== 测试 1: 使用 NerfAcc渲染 ==========
    if generator_test.use_nerfacc:
        generator_test.use_test_kwargs = False  # 使用训练模式（启用nerfacc）

        print("[测试 1] NerfAcc 渲染性能:")
        times = []
        samples_list = []

        # 预热
        with torch.no_grad():
            z = zdist.sample((batch_size,))
            _ = generator_test(z, label)

        # 正式测试
        for i in range(num_iterations):
            with torch.no_grad():
                z = zdist.sample((batch_size,))

                torch.cuda.synchronize()
                start = time.time()

                rgb, _, _, extras = generator_test(z, label)

                torch.cuda.synchronize()
                end = time.time()

                times.append(end - start)
                if 'n_samples' in extras:
                    samples_list.append(extras['n_samples'])

        avg_time = np.mean(times)
        std_time = np.std(times)
        avg_samples = np.mean(samples_list) if samples_list else 0

        print(f"  平均时间: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"  平均采样点数: {avg_samples:.0f}")

        nerfacc_time = avg_time
        nerfacc_samples = avg_samples
    else:
        print("[跳过] NerfAcc 未启用")
        nerfacc_time = None
        nerfacc_samples = None

    # ========== 测试 2: 原始渲染（不使用NerfAcc）==========
    generator_test.use_test_kwargs = True  # 使用测试模式（禁用nerfacc）

    print("\n[测试 2] 原始渲染性能 (不使用NerfAcc):")
    times = []

    # 预热
    with torch.no_grad():
        z = zdist.sample((batch_size,))
        _ = generator_test(z, label)

    # 正式测试
    for i in range(num_iterations):
        with torch.no_grad():
            z = zdist.sample((batch_size,))

            torch.cuda.synchronize()
            start = time.time()

            rgb, _, _, extras = generator_test(z, label)

            torch.cuda.synchronize()
            end = time.time()

            times.append(end - start)

    avg_time = np.mean(times)
    std_time = np.std(times)

    # 计算理论采样点数
    N_samples = config['nerf']['N_samples']
    N_rays = 4096  # ray_sampler.N_samples
    theoretical_samples = N_samples * N_rays * batch_size

    print(f"  平均时间: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"  理论采样点数: {theoretical_samples} ({N_samples} samples × {N_rays} rays × {batch_size} batch)")

    original_time = avg_time
    original_samples = theoretical_samples

    # ========== 对比分析 ==========
    print(f"\n{'='*60}")
    print("性能对比分析:")
    print(f"{'='*60}")

    if nerfacc_time is not None:
        speedup = original_time / nerfacc_time
        sample_reduction = (1 - nerfacc_samples / original_samples) * 100

        print(f"时间对比:")
        print(f"  原始渲染: {original_time*1000:.2f} ms")
        print(f"  NerfAcc:  {nerfacc_time*1000:.2f} ms")
        print(f"  加速比: {speedup:.2f}x")

        if speedup < 1.0:
            print(f"  ⚠️  警告: NerfAcc 反而更慢了 {(1/speedup - 1)*100:.1f}%!")
        elif speedup < 1.2:
            print(f"  ⚠️  警告: 加速效果不明显 (<20%)")
        else:
            print(f"  ✓ 加速效果: {(speedup-1)*100:.1f}%")

        print(f"\n采样点对比:")
        print(f"  原始渲染: {original_samples:.0f} 点")
        print(f"  NerfAcc:  {nerfacc_samples:.0f} 点")
        print(f"  减少: {sample_reduction:.1f}%")

        if sample_reduction < 10:
            print(f"  ⚠️  警告: 采样点几乎没有减少!")
            print(f"  可能原因:")
            print(f"    1. Occupancy grid 没有正确更新")
            print(f"    2. 场景过于密集，没有空白区域可跳过")
            print(f"    3. alpha_thre 设置过低 (当前: 0.0)")

    print(f"{'='*60}\n")


def test_occupancy_grid_update(config):
    """测试占据网格更新性能"""
    device = torch.device('cuda')

    # 初始化模型
    print("初始化模型...")
    generator_test, _ = build_models(config, disc=False)
    generator_test = generator_test.to(device)
    generator_test.train()

    if not generator_test.use_nerfacc:
        print("NerfAcc 未启用，跳过测试")
        return

    # 准备测试数据
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)
    batch_size = config['training']['batch_size']

    label = torch.zeros(batch_size, 3, device=device)
    z_sample = zdist.sample((1,))

    print(f"\n{'='*60}")
    print("测试 Occupancy Grid 更新性能:")
    print(f"{'='*60}\n")

    # 测试简单更新
    print("[测试 1] 简单球形估计更新:")
    torch.cuda.synchronize()
    start = time.time()

    generator_test.update_occupancy_grid(step=0)

    torch.cuda.synchronize()
    end = time.time()

    simple_time = (end - start) * 1000
    print(f"  时间: {simple_time:.2f} ms")

    # 测试网络更新
    print("\n[测试 2] 网络精确更新:")
    torch.cuda.synchronize()
    start = time.time()

    generator_test.update_occupancy_grid_with_network(step=0, label=label, z_sample=z_sample)

    torch.cuda.synchronize()
    end = time.time()

    network_time = (end - start) * 1000
    print(f"  时间: {network_time:.2f} ms")

    # 分析
    print(f"\n对比:")
    print(f"  网络更新比简单更新慢 {network_time/simple_time:.1f}x")

    if network_time > 500:
        print(f"  ⚠️  警告: 网络更新非常耗时 ({network_time:.0f} ms)")
        print(f"  建议: 增大更新间隔 (如 32 或 64 步)")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--iterations', type=int, default=10)
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    print("\n" + "="*60)
    print("NerfAcc 性能调试工具")
    print("="*60)

    # 测试渲染性能
    test_render_performance(config, num_iterations=args.iterations)

    # 测试占据网格更新性能
    test_occupancy_grid_update(config)
