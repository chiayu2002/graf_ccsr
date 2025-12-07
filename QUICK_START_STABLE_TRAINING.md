# 🚀 快速开始：稳定训练配置

## ✅ 修改完成

您的 `train.py` 已经修改完成，**不使用混合精度训练（scaler）**，只包含：

1. ✅ **梯度裁剪** - 防止梯度爆炸
2. ✅ **梯度惩罚频率控制** - 可选的加速优化
3. ✅ **降低的学习率** - 匹配 quality 配置的样本数

## 🎯 立即开始训练

```bash
python train.py --config configs/nerfacc_quality_stable.yaml
```

## 📋 配置说明

### 稳定性参数

```yaml
training:
  lr_g: 0.0004      # Generator 学习率（从 0.0008 降低 50%）
  lr_d: 0.00001     # Discriminator 学习率（从 0.00002 降低 50%）
  grad_clip: 1.0    # 梯度裁剪阈值（防止梯度爆炸）
  reg_every: 1      # 梯度惩罚频率（1=每步都计算）
```

### NerfAcc 质量参数

```yaml
nerfacc:
  resolution: 32              # Grid 分辨率（清晰，无黑线）
  render_step_size: 0.08      # 初始采样步长（密集采样）
  # alpha_thre: 0.01 (在代码中，保守裁剪)
  occ_grid_update_interval: 128  # Grid 更新频率
```

## 📊 预期结果

### 训练指标

```
[Optimization] Gradient Clipping: max_norm=1.0  ← 确认启用
Generator LR: 0.0004  ← 新学习率
Discriminator LR: 0.00001

[NerfAcc] Iter 100: Samples XXX/491520 (40-50% reduction)  ← 样本减少
```

### Loss 曲线

```
loss/generator: 稳定下降（不再上升！）
loss/discriminator: 快速下降后稳定
loss/regularizer: 正常下降
```

### 图像质量

- ✅ **无黑线**
- ✅ **清晰**，有细节
- ✅ 边缘平滑

### 训练速度

- **~48-50 it/min**（比基准 47 it/min 略快）
- 样本生成：**~2-3s**（从 7.5s 大幅改进）

## 🔧 可选优化

### 选项 1：减少梯度惩罚频率（更快）

如果想要进一步加速，可以在配置中添加：

```yaml
training:
  reg_every: 2  # 每 2 步计算一次梯度惩罚（节省 ~50% 计算）
```

**预期**：速度提升 10-15%，但可能轻微影响稳定性

### 选项 2：调整梯度裁剪阈值

如果训练仍有波动：

```yaml
training:
  grad_clip: 0.5  # 更激进的裁剪（从 1.0 降低）
```

如果训练太保守：

```yaml
training:
  grad_clip: 2.0  # 更宽松的裁剪（从 1.0 提高）
```

## ❓ 故障排查

### 问题 1：Generator loss 仍然上升

**解决方案**：进一步降低学习率

```yaml
training:
  lr_g: 0.0004 → 0.0003
  lr_d: 0.00001 → 0.000008
```

### 问题 2：训练太慢收敛

**解决方案**：轻微提高学习率（只有在训练完全稳定后）

```yaml
training:
  lr_g: 0.0004 → 0.0005
```

### 问题 3：仍有轻微黑线

**解决方案**：进一步提高 NerfAcc 质量参数

```yaml
nerfacc:
  resolution: 32 → 40
  render_step_size: 0.08 → 0.06
```

并在 `submodules/nerf_pytorch/run_nerf_mod.py:261` 修改：

```python
alpha_thre=0.01 → 0.005
```

### 问题 4：想要更高速度

**不推荐**：回到 aggressive 配置质量太差

**推荐**：使用 reg_every=2 + 稍微降低质量

```yaml
training:
  reg_every: 2
nerfacc:
  alpha_thre: 0.015  # 代码中，轻微提高
```

## 📝 代码变更摘要

### train.py 的关键变更

1. **默认 occ_grid_update_interval**: 16 → 128

2. **添加梯度裁剪支持**：
```python
grad_clip = config['training'].get('grad_clip', None)

if grad_clip is not None:
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)
    torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
```

3. **添加梯度惩罚频率控制**：
```python
reg_every = config['training'].get('reg_every', 1)

if it % reg_every == 0:
    reg = config['training']['reg_param'] * compute_grad2(d_real, rgbs).mean()
else:
    reg = torch.tensor(0.0, device=device)
```

4. **移除混合精度训练**：代码更简单，更容易理解和调试

## ✅ 准备完成

您现在可以开始训练了：

```bash
python train.py --config configs/nerfacc_quality_stable.yaml
```

**监控重点**：
1. Loss 曲线是否稳定下降
2. 样本图像是否清晰无黑线
3. 训练速度是否在 48-50 it/min

祝训练顺利！🎉
