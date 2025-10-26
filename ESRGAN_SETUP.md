# ESRGAN 集成使用指南

本项目已成功集成预训练的 RRDB_ESRGAN_x4 模型，用于提升 NeRF 生成图像的质量。

## 🎯 工作原理

1. **NeRF 渲染**：生成 64x64 的图像 patch
2. **下采样**：将图像降至 16x16（模拟低分辨率输入）
3. **ESRGAN 超分辨率**：使用预训练模型将 16x16 上采样回 64x64
4. **GAN 训练**：Discriminator 评估超分辨率后的图像质量

**关键优势**：
- ESRGAN 输出成为最终输出（真正发挥作用）
- NeRF 学习生成适合超分辨率处理的特征
- 利用 ESRGAN 的预训练知识提升图像细节

## 📥 下载预训练模型

### 方法 1：从官方仓库下载

```bash
# 创建模型目录
mkdir -p pretrained_models

# 下载 RRDB_ESRGAN_x4 模型
wget https://github.com/xinntao/ESRGAN/releases/download/v0.0.0/RRDB_ESRGAN_x4.pth \
     -O pretrained_models/RRDB_ESRGAN_x4.pth
```

### 方法 2：使用 gdown（如果 wget 失败）

```bash
pip install gdown
gdown --id 1TPrz5QKd8DHHt1k8SRtm6tMiPjz_Qene -O pretrained_models/RRDB_ESRGAN_x4.pth
```

### 方法 3：手动下载

访问 [ESRGAN GitHub Releases](https://github.com/xinntao/ESRGAN/releases)，下载 `RRDB_ESRGAN_x4.pth`，放到 `pretrained_models/` 目录下。

## ⚙️ 配置文件设置

编辑 `configs/default.yaml`：

### 使用 ESRGAN（推荐）

```yaml
esrgan:
  enabled: true
  pretrained_path: 'pretrained_models/RRDB_ESRGAN_x4.pth'
  freeze: true  # 冻结 ESRGAN 参数，只训练 NeRF
  scale_factor: 4

ccsr:
  enabled: false  # 关闭 CCSR
```

### 使用自定义 CCSR

```yaml
esrgan:
  enabled: false

ccsr:
  enabled: true
  num_views: 8
  scale_factor: 4
```

## 🚀 训练

```bash
python train.py --config configs/default.yaml
```

训练时会看到：
- `loss/generator`：GAN 生成器损失
- `loss/sr_consistency`：超分辨率一致性损失
- `loss/discriminator`：判别器损失

## 📊 工作流程详解

```
训练流程：
┌─────────────┐
│  NeRF 渲染  │ → 64x64 RGB patch
└──────┬──────┘
       │
       ↓ 下采样
┌─────────────┐
│ 16x16 LR    │
└──────┬──────┘
       │
       ↓ ESRGAN (冻结参数)
┌─────────────┐
│ 64x64 SR    │ ← 这是最终输出！
└──────┬──────┘
       │
       ↓
┌─────────────┐
│Discriminator│ → 判断真假
└─────────────┘
```

## 🔧 高级配置

### 微调 ESRGAN（不推荐初学者）

如果你的数据集很特殊，可以微调 ESRGAN：

```yaml
esrgan:
  enabled: true
  pretrained_path: 'pretrained_models/RRDB_ESRGAN_x4.pth'
  freeze: false  # 允许训练 ESRGAN
  scale_factor: 4
```

注意：这会增加训练难度和显存占用。

### 调整一致性损失权重

在 `graf/train_step.py` 中修改：

```python
def __init__(self, alpha_init=1.0, alpha_decay=0.0001):
    # alpha_init: 初始权重
    # alpha_decay: 权重衰减速度
```

## 🐛 常见问题

### Q1: 提示找不到预训练模型

**A**: 检查路径是否正确：
```bash
ls -lh pretrained_models/RRDB_ESRGAN_x4.pth
```

### Q2: 显存不足

**A**: 尝试：
- 减小 batch_size
- 减小 ray_sampler.N_samples
- 确保 `freeze: true`（冻结 ESRGAN）

### Q3: 损失不稳定

**A**:
- 降低学习率
- 检查 alpha_decay 设置
- 确认预训练模型加载成功

### Q4: 想完全移除超分辨率

**A**: 设置：
```yaml
esrgan:
  enabled: false
ccsr:
  enabled: false
```

## 📈 效果对比

训练后，你应该能看到：
- 更清晰的纹理细节
- 更少的模糊
- 更好的边缘定义

对比方式：
1. 使用 `enabled: false` 训练基线模型
2. 使用 `enabled: true` 训练 ESRGAN 增强模型
3. 比较生成的样本图像

## 📚 参考资料

- [ESRGAN 论文](https://arxiv.org/abs/1809.00219)
- [ESRGAN GitHub](https://github.com/xinntao/ESRGAN)
- [NeRF 论文](https://arxiv.org/abs/2003.08934)

## 🎉 总结

现在 ESRGAN 已经**真正发挥作用**：
- ✅ ESRGAN 输出成为最终输出
- ✅ Discriminator 评估 ESRGAN 的结果
- ✅ NeRF 学习生成适合超分辨率的特征
- ✅ 完全利用预训练模型的能力

祝训练顺利！
