# NerfAcc 性能瓶颈分析

## 测试结果

- **没有 NerfAcc**: 300分钟 → ~14000 iterations (47 it/min)
- **有 NerfAcc**: 300分钟 → ~13000 iterations (43 it/min)
- **变慢**: ~8-10%

## 🔴 主要瓶颈：Occupancy Grid 更新

### 当前实现 (`generator.py:228-281`)

```python
def update_occupancy_grid_with_network(self, step, label, z_sample=None):
    n_views = 4  # 使用 4 个随机 view 方向

    for view_dir in random_dirs:  # 对每个 view
        for i in range(0, positions.shape[0], chunk_size):  # 遍历所有 grid 点
            raw = network_query_fn(pos_chunk, view_input, ...)
```

### 计算成本

对于 **resolution=128** 的 grid：
- Grid 点数: `128³ = 2,097,152` 个点
- View 数量: `4` 个
- 每次更新总计算: `2,097,152 × 4 = 8,388,608` 次 network forward
- 更新频率: 每 `16` 步
- **每步平均开销**: `8,388,608 / 16 ≈ 524,288` 次额外 forward

### 开销对比

正常训练每步：
- Generator forward: ~1-2 次（生成图像）
- Discriminator forward: ~1 次

NerfAcc 每步平均增加：
- **524,288 次 network forward**（用于 grid 更新）
- 即使在 `no_grad()` 中，这仍然是**巨大的开销**！

## 🔴 问题 2：分 Batch 处理效率低

### 当前实现 (`run_nerf_mod.py:175-339`)

```python
for i in range(0, N_rays, batch_rays):  # 每个 batch
    # 调用一次 estimator.sampling()
    ray_indices, t_starts, t_ends = estimator.sampling(...)

    # 手动累积结果
    rgbs, sigmas = query_rgb_sigma(...)
    weights = render_weight_from_density(...)
    rgb_map = accumulate_along_rays(...)
```

### 问题

1. **多次调用 estimator.sampling()**
   - 每个 batch 都要查询 occupancy grid
   - 可能有重复的 grid 查询开销

2. **手动累积 vs 官方 API**
   - `accumulate_along_rays()` 可能不如 `nerfacc.rendering()` 高效
   - 缺少 GPU kernel 级别的优化

## 🔴 问题 3：Viewdir 处理开销

虽然影响较小，但 `vdir.mean()` 也增加了一些开销。

## ⚡ 优化方案

### 方案 1：降低 Grid 更新频率（快速修复）⭐

```yaml
# train.py 或命令行参数
--occ_grid_update_interval 256  # 从 16 改为 256
```

**预期效果**：
- 减少 16× 的 grid 更新开销
- 从每步平均 524k 降到 32k 次 forward
- 可能牺牲一些采样精度，但值得尝试

### 方案 2：降低 Grid Resolution（中等修复）

```yaml
# configs/default.yaml
nerfacc:
  resolution: 64  # 从 128 改为 64
```

**预期效果**：
- Grid 点数从 2M 降到 262k（8× 减少）
- 每次更新从 8.4M 降到 1M 次 forward
- 精度略有下降，但仍足够

### 方案 3：减少 View 数量（简单修复）

```python
# generator.py:243
n_views = 1  # 从 4 改为 1
```

**预期效果**：
- 4× 减少 grid 更新开销
- 单个 view 可能不够 robust，建议配合方案 1

### 方案 4：使用简单球形更新（最快，但不准确）

```bash
python train.py --use_simple_occ
```

**预期效果**：
- 几乎零开销的 grid 更新
- 但采样质量很差，不推荐

### 方案 5：去掉分 Batch（最优，需要重构）

直接传递所有 rays 给 estimator：

```python
# 一次性处理所有 rays
ray_indices, t_starts, t_ends = estimator.sampling(
    rays_o=all_rays_o,  # [N_rays_total, 3]
    rays_d=all_rays_d,
    sigma_fn=sigma_fn,
    ...
)
```

**预期效果**：
- 减少重复的 grid 查询
- 更好的 GPU 利用率
- 需要修改代码结构

### 方案 6：使用官方 `nerfacc.rendering()`

替换手动的 `accumulate_along_rays`：

```python
from nerfacc import rendering

color, opacity, depth, extras = rendering(
    t_starts, t_ends, ray_indices,
    n_rays=N_rays,
    rgb_sigma_fn=rgb_sigma_fn
)
```

## 📊 推荐的优化顺序

### 立即尝试（5分钟）

1. ✅ **降低更新频率**: `--occ_grid_update_interval 256`
2. ✅ **减少 view 数**: `n_views = 1`
3. ✅ **降低 resolution**: `resolution: 64`

**组合预期加速**：
- Grid 更新开销: `2M × 4 × 16` → `262k × 1 × 256`
- 减少约 **122×** 的 grid 更新开销！
- 应该能看到明显加速

### 中期优化（1-2小时）

4. ✅ 去掉分 batch 处理
5. ✅ 使用 `nerfacc.rendering()`

### 长期优化（需要重构）

6. ✅ 修复 viewdir 处理
7. ✅ 完全按照官方范例重写

## 🎯 测试计划

### 测试 1：Baseline（快速验证）

```bash
python train.py --config configs/default.yaml \
    --occ_grid_update_interval 256
```

修改 `configs/default.yaml`:
```yaml
nerfacc:
  use_nerfacc: true
  resolution: 64
```

修改 `generator.py:243`:
```python
n_views = 1
```

**预期结果**: 应该比没有 NerfAcc 快 1.5-2×

### 测试 2：逐步调优

如果测试 1 成功，逐步提高精度：
- `resolution: 64 → 96 → 128`
- `n_views: 1 → 2 → 4`
- `update_interval: 256 → 128 → 64 → 32`

找到速度和质量的最佳平衡点。

## 🔬 性能监控

在 `train.py` 中添加更详细的计时：

```python
# 监控各个阶段的时间
times = {
    'occ_update': [],
    'generator_forward': [],
    'discriminator': [],
    'total_iter': []
}
```

这样可以准确知道瓶颈在哪里。

## 结论

**当前 NerfAcc 变慢的主要原因**：

1. **Occupancy grid 更新太昂贵**（128³ × 4 views × 每 16 步）
2. 分 batch 处理可能不够高效
3. 使用了手动的渲染累积而非优化的官方 API

**最快的修复方案**：
- Resolution: 128 → 64
- Views: 4 → 1
- Update interval: 16 → 256

这应该能**立即**看到加速效果！
