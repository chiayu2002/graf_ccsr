"""
NerfAcc 性能分析工具
精确测量 occupancy grid 更新、采样、渲染各部分的时间开销
"""
import torch
import time
import argparse
import numpy as np
from graf.config import load_config, build_models, get_data

def profile_nerfacc(config_path, n_iters=100):
    """分析 NerfAcc 各部分的性能"""

    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建模型
    generator, discriminator = build_models(config, disc=True)
    generator = generator.to(device)
    generator.eval()

    # 准备数据
    dset, hwfr = get_data(config)
    zdist = torch.distributions.normal.Normal(
        torch.zeros(config['z_dist']['dim']).to(device),
        torch.ones(config['z_dist']['dim']).to(device)
    )

    # 检查是否使用 NerfAcc
    use_nerfacc = config.get('nerfacc', {}).get('use_nerfacc', False)

    print("=" * 70)
    print(f"NerfAcc 性能分析")
    print("=" * 70)
    print(f"配置文件: {config_path}")
    print(f"NerfAcc 状态: {'启用' if use_nerfacc else '禁用'}")
    print(f"迭代次数: {n_iters}")
    print(f"设备: {device}")

    if use_nerfacc:
        resolution = config['nerfacc'].get('resolution', 128)
        print(f"Grid 解析度: {resolution}³ = {resolution**3:,} 个点")

    print("=" * 70)

    # 计时器
    times = {
        'occ_grid_update': [],
        'generator_forward': [],
        'total_iter': []
    }

    # 创建 label
    num_classes = config['discriminator']['num_classes']
    label = torch.zeros(1, num_classes).to(device)

    # 模拟训练循环
    print("\n开始性能测试...")

    for it in range(n_iters):
        iter_start = time.time()

        # 1. Occupancy grid 更新（如果启用 NerfAcc）
        occ_time = 0
        if use_nerfacc and hasattr(generator, 'use_nerfacc') and generator.use_nerfacc:
            # 假设每 128 步更新一次
            if it % 128 == 0:
                torch.cuda.synchronize()
                occ_start = time.time()

                z_sample = zdist.sample((1,))
                generator.update_occupancy_grid_with_network(it, label, z_sample)

                torch.cuda.synchronize()
                occ_time = (time.time() - occ_start) * 1000  # ms
                times['occ_grid_update'].append(occ_time)

        # 2. Generator forward
        torch.cuda.synchronize()
        gen_start = time.time()

        with torch.no_grad():
            z = zdist.sample((1,))
            fake_img = generator(z, label)

        torch.cuda.synchronize()
        gen_time = (time.time() - gen_start) * 1000  # ms
        times['generator_forward'].append(gen_time)

        # 总时间
        torch.cuda.synchronize()
        iter_time = (time.time() - iter_start) * 1000  # ms
        times['total_iter'].append(iter_time)

        # 打印进度
        if (it + 1) % 10 == 0:
            avg_iter = np.mean(times['total_iter'][-10:])
            avg_gen = np.mean(times['generator_forward'][-10:])
            print(f"[{it+1}/{n_iters}] Avg iter: {avg_iter:.1f}ms, Gen: {avg_gen:.1f}ms", end='')

            if occ_time > 0:
                print(f", Occ update: {occ_time:.1f}ms")
            else:
                print()

    # 统计结果
    print("\n" + "=" * 70)
    print("性能统计")
    print("=" * 70)

    avg_iter = np.mean(times['total_iter'])
    avg_gen = np.mean(times['generator_forward'])

    print(f"\n平均每次迭代时间: {avg_iter:.2f} ms")
    print(f"  - Generator forward: {avg_gen:.2f} ms ({avg_gen/avg_iter*100:.1f}%)")

    if times['occ_grid_update']:
        avg_occ = np.mean(times['occ_grid_update'])
        occ_count = len(times['occ_grid_update'])
        avg_occ_per_iter = avg_occ * occ_count / n_iters

        print(f"  - Occ grid update: {avg_occ:.2f} ms (每次更新)")
        print(f"    * 更新次数: {occ_count}")
        print(f"    * 平均每步开销: {avg_occ_per_iter:.2f} ms ({avg_occ_per_iter/avg_iter*100:.1f}%)")

    # 预估吞吐量
    iters_per_sec = 1000 / avg_iter
    iters_per_min = iters_per_sec * 60

    print(f"\n预估吞吐量:")
    print(f"  - {iters_per_sec:.2f} iterations/sec")
    print(f"  - {iters_per_min:.1f} iterations/min")
    print(f"  - 300分钟可完成: {int(iters_per_min * 300):,} iterations")

    # 如果有 occupancy grid 更新，计算其开销占比
    if times['occ_grid_update']:
        total_occ_time = sum(times['occ_grid_update'])
        total_time = sum(times['total_iter'])
        occ_overhead_pct = (total_occ_time / total_time) * 100

        print(f"\n⚠️ Occupancy grid 更新开销: {occ_overhead_pct:.1f}%")
        if occ_overhead_pct > 20:
            print(f"   建议: 开销过大，考虑降低更新频率或 grid 分辨率")

    print("\n" + "=" * 70)

    # 内存使用
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"\nGPU 内存使用:")
        print(f"  - Allocated: {allocated:.2f} GB")
        print(f"  - Reserved: {reserved:.2f} GB")

    return times


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NerfAcc 性能分析')
    parser.add_argument('--config', default='configs/default.yaml',
                        help='配置文件路径')
    parser.add_argument('--iters', type=int, default=100,
                        help='测试迭代次数')
    args = parser.parse_args()

    profile_nerfacc(args.config, args.iters)
