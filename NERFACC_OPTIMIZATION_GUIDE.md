# NerfAcc 优化指南 - 解决变慢问题

## 🔴 问题诊断

你的测试结果显示 NerfAcc 反而变慢了：
- **没有 NerfAcc**: 300分钟 → ~14000 iterations (47 it/min)
- **有 NerfAcc**: 300分钟 → ~13000 iterations (43 it/min)
- **变慢**: ~8-10%

**根本原因**: Occupancy grid 更新的开销超过了采样节省的时间！

## 📊 开销分析

### 原始配置的开销

对于 `resolution=128`, `n_views=4`, `update_interval=16`:

```
Grid 更新计算量 = 128³ × 4 views ÷ 16 steps
                = 2,097,152 × 4 ÷ 16
                = 524,288 次 network forward / step
```

这比正常的 Generator forward (1-2次/step) 多**几十万倍**！

### 优化后的配置

对于 `resolution=64`, `n_views=1`, `update_interval=128`:

```
Grid 更新计算量 = 64³ × 1 view ÷ 128 steps
                = 262,144 × 1 ÷ 128
                = 2,048 次 network forward / step
```

**减少**: 524,288 → 2,048 = **256× 降低**！

## ✅ 已完成的优化

### 1. 减少 View 数量
**文件**: `graf/models/generator.py:243`

```python
# 之前
n_views = 4  # 使用4個隨機view方向

# 之后
n_views = 1  # 🔧 從 4 改為 1 以減少開銷
```

**效果**: 4× 减少计算量

### 2. 降低更新频率
**文件**: `train.py:74`

```python
# 之前
parser.add_argument('--occ_grid_update_interval', type=int, default=16, ...)

# 之后
parser.add_argument('--occ_grid_update_interval', type=int, default=128, ...)
```

**效果**: 8× 减少更新频率

### 3. 创建优化配置
**文件**: `configs/nerfacc_optimized.yaml`

```yaml
nerfacc:
  use_nerfacc: true
  resolution: 64  # 从 128 降到 64
```

**效果**: 8× 减少 grid 点数

## 🚀 测试步骤

### 快速测试（推荐）

直接使用优化配置训练：

```bash
python train.py --config configs/nerfacc_optimized.yaml \
    --occ_grid_update_interval 128
```

**预期结果**:
- 应该比没有 NerfAcc 快 **1.5-2×**
- 300分钟应该能完成 **~21000-28000 iterations**

### 性能分析（可选）

如果想详细了解各部分的时间开销：

```bash
# 测试优化后的配置
python profile_nerfacc.py --config configs/nerfacc_optimized.yaml --iters 100

# 对比原始配置（禁用 NerfAcc）
python profile_nerfacc.py --config configs/default.yaml --iters 100
```

这会输出：
- 每次迭代的平均时间
- Generator forward 时间
- Occupancy grid 更新时间和占比
- 预估吞吐量

## ⚙️ 参数调优

如果优化后速度仍不理想，可以进一步调整：

### 进一步降低开销

```bash
# 更低的更新频率
python train.py --config configs/nerfacc_optimized.yaml \
    --occ_grid_update_interval 256  # 从 128 提高到 256

# 或手动修改 configs/nerfacc_optimized.yaml:
nerfacc:
  resolution: 48  # 更低的分辨率（48³ = 110,592 个点）
```

### 逐步提高精度（如果速度足够）

一旦看到加速效果，可以逐步提高精度：

```yaml
# Step 1: 基础优化（当前）
resolution: 64
n_views: 1
update_interval: 128

# Step 2: 提高 grid 精度
resolution: 96
n_views: 1
update_interval: 128

# Step 3: 增加 view 数
resolution: 96
n_views: 2
update_interval: 128

# Step 4: 提高更新频率
resolution: 96
n_views: 2
update_interval: 64

# Step 5: 接近原始配置
resolution: 128
n_views: 2
update_interval: 64
```

每次调整后测试速度，找到最佳平衡点。

## 📈 监控指标

训练时关注这些指标：

### 1. 速度指标
- **iterations/min**: 应该提高到 ~70-90 it/min（vs 原来的 43）
- **Occ grid update time**: 应该 < 总时间的 10%

### 2. 质量指标
- **采样点减少率**: 50-80% 是健康的（不是越高越好）
- **生成图像质量**: 不应该因为优化而明显下降

### 3. WandB 日志
```
nerfacc/samples_ratio: 应该在 0.2-0.5 之间
nerfacc/occ_grid_step: 监控更新次数
```

## 🎯 预期效果

### 最保守优化 (resolution=64, n_views=1, interval=128)

```
开销减少: 256×
预期加速: 1.5-2×
吞吐量: ~70-90 it/min
300分钟完成: ~21000-27000 iterations
```

### 激进优化 (resolution=48, n_views=1, interval=256)

```
开销减少: ~600×
预期加速: 2-2.5×
吞吐量: ~90-120 it/min
300分钟完成: ~27000-36000 iterations
```

但可能牺牲一些采样精度。

## 🔧 故障排查

### 如果仍然很慢

1. **检查是否真的启用了优化**
   ```python
   # 确认 generator.py:243 是 n_views=1
   # 确认启动时传入了 --occ_grid_update_interval 128
   ```

2. **使用性能分析工具**
   ```bash
   python profile_nerfacc.py --config configs/nerfacc_optimized.yaml
   ```
   查看 "Occ grid update" 的占比，应该 < 10%

3. **临时禁用 grid 更新测试**
   ```bash
   # 使用简单球形估计（几乎零开销）
   python train.py --config configs/nerfacc_optimized.yaml --use_simple_occ
   ```
   如果这样快了，说明确实是 grid 更新的问题

### 如果质量下降

1. **逐步提高 resolution**: 64 → 80 → 96
2. **增加 view 数量**: 1 → 2
3. **降低 update interval**: 128 → 64

## 📝 总结

**关键修改**:
1. ✅ `n_views: 4 → 1` (generator.py)
2. ✅ `update_interval: 16 → 128` (train.py)
3. ✅ `resolution: 128 → 64` (nerfacc_optimized.yaml)

**总效果**: 减少约 256× 的 grid 更新开销

**立即测试**:
```bash
python train.py --config configs/nerfacc_optimized.yaml
```

**预期**: NerfAcc 现在应该能带来 **1.5-2× 的加速**，而不是变慢！

---

如有问题，查看详细分析: `analyze_nerfacc_overhead.md`
