# NerfAcc 速度 vs 质量调优指南

## 📊 当前性能分析

从最新测试结果：
- **当前速度**: 40 it/min (1.5s/it)
- **基准速度** (无 NerfAcc): 47 it/min
- **目标**: 100-150 it/min

**问题**: NerfAcc 优化后仍比基准慢，说明 NerfAcc 开销超过了节省的计算量。

## 🎯 解决方案：激进优化配置

已创建新配置文件 `configs/nerfacc_aggressive.yaml`，采用更激进的参数：

### 激进参数对比

| 参数 | 保守 (optimized) | 激进 (aggressive) | 影响 |
|------|------------------|-------------------|------|
| `resolution` | 24 | **16** | Grid 查询速度 3.4× 提升 |
| `render_step_size` | 0.12 | **0.2** | 初始样本数减少 40% |
| `alpha_thre` | 0.03 | **0.05** | 样本裁剪从 65% → 75-80% |

### 预期效果

**速度提升**:
- estimator.sampling: 75ms → **30-40ms**
- 样本减少: 65% → **75-80%**
- **整体速度: 80-120 it/min** (2-3× 加速)

**质量影响**:
- 可能会有轻微的细节损失
- 如果质量不可接受，可以按下文调整参数

## 🚀 测试步骤

### 1️⃣ 测试激进配置

```bash
python train.py --config configs/nerfacc_aggressive.yaml
```

**观察指标**:
```
[DEBUG] estimator.sampling took XX.XXms  ← 应该 < 50ms
[NerfAcc] Iter X: Samples XXX/819200 (XX.X% reduction)  ← 应该 > 75%
Epoch 0: XXX/180 [XX:XX<XX:XX, X.XXs/it]  ← 应该 < 1.0s/it
```

**期望结果**: 速度 > 60 it/min (1.0s/it)

### 2️⃣ 如果速度仍不理想

尝试**更激进**的参数（进一步降低质量以换取速度）：

```bash
python train.py --config configs/nerfacc_aggressive.yaml \
    --occ_grid_update_interval 256  # 减少 grid 更新频率 (128 → 256)
```

或者修改 `configs/nerfacc_aggressive.yaml`:
```yaml
nerfacc:
  resolution: 12              # 从 16 降低到 12 (极快)
  render_step_size: 0.25      # 从 0.2 提高到 0.25 (更大步长)
```

### 3️⃣ 如果质量下降太多

**逐步调整**参数以平衡速度和质量：

#### 选项 A: 轻微提高质量（牺牲 10-20% 速度）

```yaml
nerfacc:
  resolution: 20              # 16 → 20
  render_step_size: 0.15      # 0.2 → 0.15
```

并在 `submodules/nerf_pytorch/run_nerf_mod.py:261` 修改：
```python
alpha_thre=0.03,  # 从 0.05 降低到 0.03
```

#### 选项 B: 显著提高质量（牺牲 30-40% 速度）

使用原始的 `configs/nerfacc_optimized.yaml`:
```bash
python train.py --config configs/nerfacc_optimized.yaml
```

这会使用更保守的参数：
- resolution: 24
- render_step_size: 0.12
- alpha_thre: 0.03 (在代码中)

## 📈 性能基准参考

| 配置 | 预期速度 | 样本减少 | 质量 | 适用场景 |
|------|----------|----------|------|----------|
| **aggressive** | 80-120 it/min | 75-80% | ⭐⭐⭐ | 快速实验、早期训练 |
| **optimized** | 50-70 it/min | 65-70% | ⭐⭐⭐⭐ | 平衡速度和质量 |
| **baseline** (无 NerfAcc) | 47 it/min | 0% | ⭐⭐⭐⭐⭐ | 最高质量 |

## 🔧 参数调优速查表

### resolution (Occupancy Grid 分辨率)

- **12**: 超快，但质量可能较差
- **16**: 激进优化，适合快速迭代
- **20**: 平衡选择
- **24**: 保守优化，较好质量
- **32+**: 接近原始质量，速度提升有限

**调整建议**: resolution³ = 网格点数，每次调整 4-8 的倍数

### render_step_size (初始采样步长)

- **0.25+**: 非常粗糙，样本极少
- **0.2**: 激进优化
- **0.15**: 平衡选择
- **0.12**: 保守优化
- **0.08**: 接近原始采样

**调整建议**: 步长越大，初始样本越少，速度越快

### alpha_thre (不透明度裁剪阈值)

在 `submodules/nerf_pytorch/run_nerf_mod.py:261` 修改：

- **0.05**: 激进裁剪，75-80% 减少
- **0.03**: 平衡裁剪，65-70% 减少
- **0.02**: 保守裁剪，50-60% 减少
- **0.01**: 轻微裁剪，40-50% 减少
- **0.001**: 几乎不裁剪

**调整建议**: 阈值越高，裁剪越多，速度越快，但可能丢失细节

### occ_grid_update_interval (Grid 更新频率)

命令行参数 `--occ_grid_update_interval N`:

- **128**: 默认值，平衡更新频率
- **256**: 减少一半更新，更快
- **512**: 极少更新，最快但可能不准确
- **64**: 频繁更新，更准确但更慢

**调整建议**: 早期训练可以用较大值（256+），后期降低到 128 提高精度

## ⚠️ 故障排查

### 问题 1: 速度没有提升

**可能原因**:
1. Grid 分辨率仍太高 → 降低到 12-16
2. alpha_thre 太低 → 提高到 0.05-0.08
3. 其他瓶颈（Discriminator 计算等）

**诊断方法**:
```bash
# 观察 estimator.sampling 时间和样本减少百分比
# 如果 sampling < 50ms 且减少 > 70%，说明 NerfAcc 已优化到位
# 速度仍慢则是其他瓶颈
```

### 问题 2: 质量下降明显（黑色、模糊、伪影）

**解决方案**:
1. 提高 resolution: 16 → 20 → 24
2. 降低 render_step_size: 0.2 → 0.15 → 0.12
3. 降低 alpha_thre: 0.05 → 0.03 → 0.02
4. 降低 occ_grid_update_interval: 256 → 128 → 64

### 问题 3: OOM (显存不足)

**解决方案**:
1. 降低 batch_size (configs 中的 `training.batch_size`)
2. 降低 chunk size (configs 中的 `training.chunk`)
3. Sigma function 已有分块处理，应该不会再 OOM

## 📝 建议的测试流程

1. **先测试激进配置**:
   ```bash
   python train.py --config configs/nerfacc_aggressive.yaml
   ```
   运行 500-1000 iterations，观察速度和样本质量

2. **评估质量**:
   - 查看 wandb 的样本图像
   - 如果质量可接受 → 继续训练
   - 如果质量太差 → 进入步骤 3

3. **调整参数**（如果需要）:
   - 根据上面的"参数调优速查表"微调
   - 每次只调整 1-2 个参数
   - 重新测试 500 iterations

4. **确定最终配置**:
   - 找到速度和质量的最佳平衡点
   - 记录参数设置
   - 开始完整训练

## 💡 推荐的起始点

**如果优先考虑速度**（快速实验）:
```bash
python train.py --config configs/nerfacc_aggressive.yaml
```

**如果优先考虑质量**（最终训练）:
```bash
python train.py --config configs/nerfacc_optimized.yaml
```

**如果不确定**（先试试激进，不行再降级）:
```bash
# 第一步：激进配置测试
python train.py --config configs/nerfacc_aggressive.yaml

# 如果质量不满意，修改配置：
# - resolution: 16 → 20
# - render_step_size: 0.2 → 0.15
# - alpha_thre: 0.05 → 0.03 (在代码中)
```

## 🎓 理解参数权衡

| 提高质量 ← → 提高速度 |
|----------------------|
| resolution: 32 → 24 → 20 → 16 → 12 |
| render_step_size: 0.08 → 0.12 → 0.15 → 0.2 → 0.25 |
| alpha_thre: 0.01 → 0.02 → 0.03 → 0.05 → 0.08 |

**核心思想**: NerfAcc 通过跳过空白区域加速，参数越激进，跳过越多，速度越快，但可能误跳有用信息导致质量下降。
