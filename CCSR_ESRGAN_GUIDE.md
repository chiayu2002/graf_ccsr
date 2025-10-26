# CCSR-ESRGAN 混合模型使用指南

## 🎯 什么是 CCSR-ESRGAN？

**CCSR-ESRGAN** 是一个混合超分辨率模型，结合了：
- **CCSR** (Consistency-Controlled Super-Resolution) 的多视角一致性机制
- **ESRGAN** (Enhanced Super-Resolution GAN) 的强大超分辨率能力

这是**最推荐的配置**，因为它同时利用了两者的优势！

---

## 🏗️ 架构对比

### 三种超分辨率方案

| 模型 | 优势 | 劣势 | 适用场景 |
|------|------|------|---------|
| **CCSR-ESRGAN**<br>(混合模型) | ✅ 多视角一致性<br>✅ 强大的 SR 能力<br>✅ 预训练知识 | ⚠️ 参数较多 | **最推荐**<br>多视角 NeRF 生成 |
| **独立 ESRGAN** | ✅ 强大的 SR 能力<br>✅ 预训练知识 | ❌ 无视角一致性 | 单视角或通用超分辨率 |
| **简单 CCSR** | ✅ 多视角一致性<br>✅ 轻量级 | ❌ SR 能力有限 | 资源受限环境 |

---

## 🔧 CCSR-ESRGAN 架构详解

```
输入: LR 图像 [B, 3, 16, 16]
  │
  ├─ CCLC (Consistency-Controlling Latent Code)
  │   └─ 为每个视角学习独特的潜在代码 [3, 64, 64]
  │
  ├─ 融合层
  │   └─ LR 图像 + 潜在代码 → [B, 6, 64, 64]
  │
  ├─ RRDB Trunk (16 RRDB blocks)
  │   └─ 强大的特征提取（来自 ESRGAN 预训练模型）
  │
  ├─ 上采样层 (x4)
  │   └─ 64x64 → 256x256
  │
  ├─ 高分辨率卷积
  │   └─ 精细化输出
  │
  └─ CEM (Consistency-Enforcing Module)
      └─ 确保 SR 输出与 LR 输入一致
        ↓
输出: SR 图像 [B, 3, 64, 64]
```

---

## 📊 关键组件

### 1. **CCLC** (Consistency-Controlling Latent Code)
- 为每个视角（0-7）学习一个独特的潜在代码
- 确保不同视角渲染的一致性
- **可训练** - 随训练动态优化

### 2. **RRDB Trunk** (来自 ESRGAN)
- 16 个 Residual-in-Residual Dense Blocks
- 可加载 ESRGAN 预训练权重
- **可冻结** - 保留预训练知识，节省训练时间

### 3. **CEM** (Consistency-Enforcing Module)
- 确保超分辨率图像下采样后与原始 LR 图像一致
- 使用模糊核 + 残差修正
- **总是可训练**

---

## ⚙️ 配置说明

### 推荐配置 (configs/default.yaml)

```yaml
ccsr_esrgan:
  enabled: true                 # 启用 CCSR-ESRGAN
  num_views: 8                  # 视角数量（匹配数据集）
  num_rrdb_blocks: 16           # RRDB blocks 数量
  nf: 64                        # 特征图通道数
  gc: 32                        # RRDB growth channel
  pretrained_path: 'pretrained_models/RRDB_ESRGAN_x4.pth'
  freeze_rrdb: true             # 冻结 RRDB（推荐）
  scale_factor: 4

# 关闭其他超分辨率模型
esrgan:
  enabled: false

ccsr:
  enabled: false
```

### 配置选项详解

#### `num_rrdb_blocks`
- **16 blocks**: 快速训练，较少参数，适合大多数场景
- **23 blocks**: 完整 ESRGAN，更强性能，但训练慢

#### `freeze_rrdb`
- **true** (推荐): 冻结 RRDB trunk，只训练 CCLC 和 CEM
  - 优势：训练快，显存少，利用预训练知识
  - 劣势：RRDB 无法适应特定数据
- **false**: 全部训练
  - 优势：RRDB 可适应特定数据
  - 劣势：训练慢，显存大，可能过拟合

#### `nf` (特征通道数)
- **64**: 标准配置，平衡性能和速度
- **32**: 轻量级，适合显存受限
- **128**: 高性能，需要更多显存

---

## 🚀 使用步骤

### 1. 下载预训练模型

```bash
bash download_esrgan.sh
```

或手动下载：
```bash
mkdir -p pretrained_models
wget https://github.com/xinntao/ESRGAN/releases/download/v0.0.0/RRDB_ESRGAN_x4.pth \
     -O pretrained_models/RRDB_ESRGAN_x4.pth
```

### 2. 确认配置

编辑 `configs/default.yaml`，确保：
- `ccsr_esrgan.enabled: true`
- `esrgan.enabled: false`
- `ccsr.enabled: false`

### 3. 开始训练

```bash
python train.py --config configs/default.yaml
```

### 4. 监控训练

在 wandb 中查看：
- `loss/sr_consistency`: SR 一致性损失
- `loss/generator`: GAN 生成器损失
- `sample/rgb`: 生成样本

---

## 📈 训练策略

### 策略 A: 快速冻结训练（推荐初学者）

```yaml
ccsr_esrgan:
  num_rrdb_blocks: 16
  freeze_rrdb: true
```

- 训练时间：最短
- 显存占用：最少
- 适合：初次尝试，资源有限

### 策略 B: 部分微调

```yaml
ccsr_esrgan:
  num_rrdb_blocks: 16
  freeze_rrdb: false
```

- 训练时间：中等
- 显存占用：中等
- 适合：有一定算力，想适应特定数据

### 策略 C: 完整训练

```yaml
ccsr_esrgan:
  num_rrdb_blocks: 23
  freeze_rrdb: false
```

- 训练时间：最长
- 显存占用：最大
- 适合：充足算力，追求极致性能

---

## 💡 优势总结

相比独立 ESRGAN：
- ✅ **多视角一致性**: CCLC 确保不同视角的协调
- ✅ **视角特定优化**: 每个视角有独特的潜在代码
- ✅ **一致性强制**: CEM 确保 SR 输出合理

相比简单 CCSR：
- ✅ **强大的 SR 能力**: 16 个 RRDB blocks
- ✅ **预训练知识**: 利用 ImageNet 等大规模数据
- ✅ **更好的细节**: 纹理、边缘恢复能力强

---

## 🐛 常见问题

### Q1: 显存不足怎么办？

**A**: 尝试以下方法：
1. 减少 `num_rrdb_blocks` (16 → 8)
2. 减少 `nf` (64 → 32)
3. 确保 `freeze_rrdb: true`
4. 减少 batch_size

### Q2: RRDB 权重没有加载成功？

**A**: 检查：
1. 预训练模型路径是否正确
2. 查看训练日志：应显示 "成功加载 XX 个预训练 RRDB 权重"
3. 如果失败，会使用随机初始化（性能会下降）

### Q3: 如何知道模型在工作？

**A**: 检查：
1. 训练日志：`使用 CCSR-ESRGAN 混合模型`
2. wandb: 应该有 `loss/sr_consistency`
3. 生成的样本应该比纯 NeRF 更清晰

### Q4: freeze_rrdb 时哪些参数在训练？

**A**:
- ✅ CCLC 潜在代码（每个视角）
- ✅ 融合卷积层
- ✅ CEM 模块
- ❌ RRDB trunk（冻结）
- ✅ 上采样层
- ✅ HR 卷积层

---

## 📚 进阶调优

### 调整视角数量

如果你的数据集有 16 个视角：
```yaml
ccsr_esrgan:
  num_views: 16
```

### 调整一致性损失权重

在 `graf/train_step.py` 中:
```python
class CCSRNeRFLoss(nn.Module):
    def __init__(self, alpha_init=1.0, alpha_decay=0.0001):
        # alpha_init: 增大 → 更重视一致性
        # alpha_decay: 增大 → 一致性权重衰减更快
```

### 使用更少的 blocks

快速原型设计：
```yaml
ccsr_esrgan:
  num_rrdb_blocks: 8  # 更少的参数
```

---

## 🎉 总结

**CCSR-ESRGAN 是最佳选择**，因为：
1. 结合两者优势
2. 预训练权重加速收敛
3. 多视角一致性
4. 灵活的训练策略

开始使用：
```bash
bash download_esrgan.sh
python train.py --config configs/default.yaml
```

祝训练顺利！ 🚀
