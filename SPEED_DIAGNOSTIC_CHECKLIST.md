# NerfAcc 速度问题诊断清单

## 🔍 已完成的检查

### ✅ 1. 验证修复已应用
```bash
grep -n "estimator.sampling" submodules/nerf_pytorch/run_nerf_mod.py
```
**结果**: ✓ 只有 1 次调用（Line 192），修复已正确应用

### ✅ 2. 验证配置启用 NerfAcc
```bash
grep "use_nerfacc" configs/nerfacc_optimized.yaml
```
**结果**: ✓ `use_nerfacc: true`，已启用

### ✅ 3. 添加调试代码
```bash
python debug_nerfacc.py
```
**结果**: ✓ 调试代码已注入到 run_nerf_mod.py

## 🚀 下一步诊断步骤

### 步骤 1: 运行带调试输出的训练

```bash
python train.py --config configs/nerfacc_optimized.yaml
```

**观察前 5 次迭代的输出**，应该看到：

```
[DEBUG] render_nerfacc called (#1), bs=8, N_rays=...
[DEBUG] estimator.sampling called (#1), rays_o.shape=torch.Size([32768, 3])
[DEBUG] estimator.sampling took XX.XX ms, returned XXXXX samples
```

**检查点**：
- [ ] `render_nerfacc` 是否被调用？（如果没有，说明在用原始render）
- [ ] `render (original)` 是否被调用？（如果是，说明NerfAcc未启用）
- [ ] `estimator.sampling` 每步只调用 1 次？（不是 8 次）
- [ ] `estimator.sampling` 的时间是多少？（应该 < 50ms）

### 步骤 2: 分析输出结果

#### 情况 A: 看到 "render (original) called"
**问题**: NerfAcc 没有被使用
**可能原因**:
1. 在测试模式而不是训练模式
2. `self.use_nerfacc` 为 False

**解决方案**:
```bash
# 检查 generator 初始化
grep -A 10 "self.use_nerfacc" graf/models/generator.py
```

#### 情况 B: 看到 "render_nerfacc called" 但速度仍然慢
**问题**: NerfAcc 在运行，但有其他瓶颈

**需要检查**:
1. `estimator.sampling` 的时间
2. Occupancy grid 更新的频率
3. 其他训练步骤的时间

**进一步诊断**:
```bash
# 运行单独的渲染速度测试
python test_nerfacc_speed.py --config configs/nerfacc_optimized.yaml --iters 100
```

### 步骤 3: 详细性能分析

如果步骤 1-2 都正常，但速度仍然慢，运行详细分析：

```python
# 在 train.py 的训练循环中添加计时
import time

for it in range(max_iterations):
    iter_start = time.time()

    # ... discriminator update ...
    torch.cuda.synchronize()
    d_time = (time.time() - iter_start) * 1000

    # ... generator update ...
    torch.cuda.synchronize()
    g_time = (time.time() - d_time_start) * 1000

    # ... occupancy grid update ...
    if it % args.occ_grid_update_interval == 0:
        torch.cuda.synchronize()
        occ_time = (time.time() - occ_start) * 1000
        print(f"Occ grid update: {occ_time:.1f}ms")

    torch.cuda.synchronize()
    total_time = (time.time() - iter_start) * 1000

    if it % 10 == 0:
        print(f"Iter {it}: Total={total_time:.1f}ms, D={d_time:.1f}ms, G={g_time:.1f}ms")
```

## 📊 预期性能指标

### 有 NerfAcc（修复后）
- **Generator forward**: 20-40 ms
- **Discriminator forward**: 10-20 ms
- **Occ grid update**: < 100 ms（每 128 步）
- **Total iteration**: 40-70 ms
- **吞吐量**: ~90-130 it/min

### 无 NerfAcc（baseline）
- **Generator forward**: 50-80 ms
- **Discriminator forward**: 10-20 ms
- **Total iteration**: 80-120 ms
- **吞吐量**: ~50-75 it/min

## 🔧 可能的问题和解决方案

### 问题 1: render (original) 被调用而不是 render_nerfacc

**检查**:
```python
# 在 graf/models/generator.py 的 __call__ 方法中
print(f"use_nerfacc={self.use_nerfacc}, use_test_kwargs={self.use_test_kwargs}")
```

**修复**:
确保训练时 `self.use_test_kwargs = False`

### 问题 2: estimator.sampling 很慢（> 100ms）

**可能原因**:
- Grid resolution 太高
- Alpha threshold 太低（导致查询太多点）
- sigma_fn 调用太多次

**修复**:
```yaml
# configs/nerfacc_optimized.yaml
nerfacc:
  resolution: 48  # 降低到 48
  alpha_thre: 0.01  # 提高阈值
```

### 问题 3: Occupancy grid 更新太频繁

**检查**:
```bash
# 确认更新间隔
python train.py --config configs/nerfacc_optimized.yaml 2>&1 | grep "occ_grid_update_interval"
```

**应该看到**: `occ_grid_update_interval: 128`（不是 16）

### 问题 4: 实际上在用 CPU 而不是 GPU

**检查**:
```python
print(f"Device: {next(generator.parameters()).device}")
```

**应该看到**: `cuda:0`（不是 cpu）

## 📝 收集诊断信息

请运行以下命令并提供输出：

```bash
# 1. 验证修复
echo "=== 检查 estimator.sampling 调用次数 ==="
grep -c "estimator.sampling" submodules/nerf_pytorch/run_nerf_mod.py

# 2. 检查配置
echo "=== 检查 NerfAcc 配置 ==="
grep -A 5 "nerfacc:" configs/nerfacc_optimized.yaml

# 3. 运行调试训练（前 20 步）
echo "=== 运行调试训练 ==="
timeout 60 python train.py --config configs/nerfacc_optimized.yaml 2>&1 | head -100

# 4. 测试单独渲染速度
echo "=== 测试渲染速度 ==="
python test_nerfacc_speed.py --config configs/nerfacc_optimized.yaml --iters 20
```

## 🎯 关键问题

请回答以下问题：

1. 运行训练时，是否看到 `[DEBUG] render_nerfacc called` 输出？
   - [ ] 是
   - [ ] 否（看到 `[DEBUG] render (original) called`）
   - [ ] 完全没有看到任何 DEBUG 输出

2. 如果看到 `render_nerfacc called`，`estimator.sampling` 的时间是多少？
   - 第1次: ___ ms
   - 第2次: ___ ms
   - 第3次: ___ ms

3. 每次迭代的总时间是多少？
   - 平均: ___ ms
   - 吞吐量: ___ it/min

4. 是否看到 "Occ grid update" 的输出？
   - [ ] 是，每 ___ 步更新一次
   - [ ] 否

## 🔄 移除调试代码

诊断完成后，移除调试代码：

```bash
python debug_nerfacc.py remove
```

---

**下一步**: 根据上述诊断结果，我们可以精确定位瓶颈所在。
