"""
性能分析工具：测量训练循环中各个部分的实际时间

使用方法：
将此脚本的代码片段插入 train.py 的训练循环中
"""

import time
import torch

# 在训练循环开始前初始化
profiler_times = {
    'generator_forward_d': [],
    'discriminator_forward_backward': [],
    'generator_forward_g': [],
    'discriminator_forward_g': [],
    'generator_backward': [],
    'other': [],
    'total': []
}

# 在 train.py 的主循环中添加计时代码
# 示例：

def profile_training_iteration():
    """
    将这些代码片段插入 train.py 的对应位置
    """
    code_snippet = '''
# ========== 在每个 iteration 开始 ==========
iter_start = time.time()

# ========== Discriminator Update ==========
# Line 244: 在 with torch.no_grad() 之前
gen_forward_d_start = time.time()

with torch.no_grad():
    x_fake, _ = generator(z, label)

gen_forward_d_time = time.time() - gen_forward_d_start

# Line 245-253: Discriminator forward + backward
disc_fb_start = time.time()

x_fake.requires_grad_()
d_fake, _ = discriminator(x_fake, label)
dloss_fake = compute_loss(d_fake, 0)
total_d_loss = dloss_real + dloss_fake + reg
total_d_loss.backward()
d_optimizer.step()

disc_fb_time = time.time() - disc_fb_start

# ========== Generator Update ==========
# Line 267: Generator forward
gen_forward_g_start = time.time()

x_fake, _, ccsr_output = generator(z, label, return_ccsr_output=True)

gen_forward_g_time = time.time() - gen_forward_g_start

# Line 268: Discriminator forward (for generator)
disc_forward_g_start = time.time()

d_fake, label_fake = discriminator(x_fake, label)

disc_forward_g_time = time.time() - disc_forward_g_start

# Line 280-285: Generator backward
gen_backward_start = time.time()

gloss = compute_loss(d_fake, 1)
if config['ccsr']['enabled']:
    ccsr_consistency_loss = ccsr_nerf_loss(ccsr_output['hr_output'], x_fake)
    gloss_all = gloss + 0.1 * ccsr_consistency_loss
else:
    gloss_all = gloss
    ccsr_consistency_loss = 0.0

gloss_all.backward()
g_optimizer.step()

gen_backward_time = time.time() - gen_backward_start

# ========== Total ==========
iter_total_time = time.time() - iter_start

# ========== 记录 ==========
if it % 10 == 0:
    other_time = iter_total_time - (gen_forward_d_time + disc_fb_time +
                                     gen_forward_g_time + disc_forward_g_time +
                                     gen_backward_time)

    profiler_times['generator_forward_d'].append(gen_forward_d_time)
    profiler_times['discriminator_forward_backward'].append(disc_fb_time)
    profiler_times['generator_forward_g'].append(gen_forward_g_time)
    profiler_times['discriminator_forward_g'].append(disc_forward_g_time)
    profiler_times['generator_backward'].append(gen_backward_time)
    profiler_times['other'].append(other_time)
    profiler_times['total'].append(iter_total_time)

# ========== 每 100 步打印统计 ==========
if it % 100 == 0 and it > 0:
    import numpy as np
    print("\\n" + "="*60)
    print(f"[Profiler] Iteration {it} - 时间分布统计 (最近 10 次平均)")
    print("="*60)

    for key in ['generator_forward_d', 'discriminator_forward_backward',
                'generator_forward_g', 'discriminator_forward_g',
                'generator_backward', 'other', 'total']:
        if profiler_times[key]:
            avg_time = np.mean(profiler_times[key][-10:]) * 1000
            total_avg = np.mean(profiler_times['total'][-10:]) * 1000
            pct = (avg_time / total_avg * 100) if total_avg > 0 else 0
            print(f"{key:30s}: {avg_time:6.1f}ms ({pct:5.1f}%)")

    total_avg = np.mean(profiler_times['total'][-10:])
    print(f"\\n预估速度: {60/total_avg:.1f} it/min")
    print("="*60 + "\\n")
    '''

    return code_snippet

if __name__ == "__main__":
    print("=" * 70)
    print("训练循环性能分析工具")
    print("=" * 70)
    print()
    print("此工具提供了需要插入 train.py 的代码片段")
    print()
    print("使用步骤：")
    print("1. 复制下面的代码片段")
    print("2. 在 train.py 的对应位置插入计时代码")
    print("3. 运行训练查看实际时间分布")
    print()
    print("=" * 70)
    print()

    snippet = profile_training_iteration()
    print(snippet)
    print()
    print("=" * 70)
    print()
    print("预期输出示例：")
    print()
    print("""
============================================================
[Profiler] Iteration 100 - 时间分布统计 (最近 10 次平均)
============================================================
generator_forward_d           :   45.2ms ( 3.8%)
discriminator_forward_backward:  542.1ms (45.9%)
generator_forward_g           :   43.8ms ( 3.7%)
discriminator_forward_g       :  102.3ms ( 8.7%)
generator_backward            :  398.7ms (33.8%)
other                         :   48.9ms ( 4.1%)
total                         : 1181.0ms (100.0%)

预估速度: 50.8 it/min
============================================================
""")
    print()
    print("如果看到：")
    print("- generator_forward (渲染) < 50ms ✓ NerfAcc 工作正常")
    print("- discriminator + generator backward > 900ms → 主要瓶颈在这里")
    print("- 这证明了 NerfAcc 已经优化到位，继续优化收益有限")
