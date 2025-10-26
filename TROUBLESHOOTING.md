# 故障排除指南 (Troubleshooting Guide)

## 🐛 常见错误和解决方案

---

### ❌ ImportError: cannot import name 'RRDB' from 'graf.models.esrgan_model'

**错误信息**:
```
ImportError: cannot import name 'RRDB' from 'graf.models.esrgan_model'
```

**原因**:
- Python 模块导入问题
- 缓存的 .pyc 文件过期
- 缺少 `__init__.py` 文件

**解决方案**:

#### 方法 1: 清除 Python 缓存（最常见）

```bash
# 删除所有 __pycache__ 目录
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# 删除所有 .pyc 文件
find . -name "*.pyc" -delete

# 重新运行
python train.py --config configs/default.yaml
```

#### 方法 2: 重新拉取最新代码

```bash
# 拉取最新更改
git pull origin claude/investigate-ccsr-function-011CUVLBpJek4ehRzrQX3dSN

# 清除缓存
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# 重新运行
python train.py --config configs/default.yaml
```

#### 方法 3: 检查文件完整性

确认以下文件存在并包含正确内容：

```bash
# 检查 esrgan_model.py 是否有 __all__
grep "__all__" graf/models/esrgan_model.py

# 应该显示:
# __all__ = ['ResidualDenseBlock', 'RRDB', 'RRDBNet', 'ESRGANWrapper']

# 检查 __init__.py 是否存在
ls -l graf/models/__init__.py
```

#### 方法 4: 验证 RRDB 类定义

```bash
# 检查 RRDB 类是否在文件中
grep "^class RRDB" graf/models/esrgan_model.py

# 应该显示:
# class RRDB(nn.Module):
```

#### 方法 5: Python 路径问题

确保你在项目根目录运行：

```bash
# 确认当前目录
pwd
# 应该显示: /path/to/graf_ccsr

# 如果不在根目录，切换到根目录
cd /path/to/graf_ccsr

# 重新运行
python train.py --config configs/default.yaml
```

---

### ❌ 显存不足 (Out of Memory)

**错误信息**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:

#### 方法 1: 减小 batch size
```yaml
# configs/default.yaml
training:
  batch_size: 4  # 从 8 降到 4 或更小
```

#### 方法 2: 使用更少的 RRDB blocks
```yaml
# configs/default.yaml
ccsr_esrgan:
  num_rrdb_blocks: 8  # 从 16 降到 8
```

#### 方法 3: 冻结 RRDB trunk
```yaml
# configs/default.yaml
ccsr_esrgan:
  freeze_rrdb: true  # 确保为 true
```

#### 方法 4: 减小特征通道数
```yaml
# configs/default.yaml
ccsr_esrgan:
  nf: 32  # 从 64 降到 32
```

#### 方法 5: 减少采样点数
```yaml
# configs/default.yaml
ray_sampler:
  N_samples: 2048  # 从 4096 降到 2048
```

---

### ❌ 预训练模型加载失败

**错误信息**:
```
警告: 预训练模型不存在: pretrained_models/RRDB_ESRGAN_x4.pth
```

**解决方案**:

#### 下载预训练模型

```bash
# 使用下载脚本
bash download_esrgan.sh

# 或手动下载
mkdir -p pretrained_models
wget https://github.com/xinntao/ESRGAN/releases/download/v0.0.0/RRDB_ESRGAN_x4.pth \
     -O pretrained_models/RRDB_ESRGAN_x4.pth

# 验证文件
ls -lh pretrained_models/RRDB_ESRGAN_x4.pth
# 应该显示约 65MB 的文件
```

#### 如果下载失败

可以暂时禁用预训练模型（性能会下降）：

```yaml
# configs/default.yaml
ccsr_esrgan:
  pretrained_path: null  # 设为 null
```

---

### ❌ 训练损失不下降

**症状**:
- 损失值很大且不变化
- 生成的图像全黑或全白
- wandb 显示 NaN 值

**解决方案**:

#### 方法 1: 检查学习率
```yaml
# configs/default.yaml
training:
  lr_g: 0.0006  # 生成器学习率
  lr_d: 0.0001  # 判别器学习率
```

如果损失爆炸，降低学习率：
```yaml
training:
  lr_g: 0.0003  # 减半
  lr_d: 0.00005
```

#### 方法 2: 检查数据

```python
# 在 train.py 中添加调试代码
print(f"x_real shape: {x_real.shape}")
print(f"x_real range: [{x_real.min():.3f}, {x_real.max():.3f}]")
print(f"label: {label}")
```

确保：
- 图像范围在 [-1, 1]
- 数据加载正常

#### 方法 3: 从简单配置开始

暂时禁用超分辨率，测试基础 NeRF：

```yaml
# configs/default.yaml
ccsr_esrgan:
  enabled: false
esrgan:
  enabled: false
ccsr:
  enabled: false
```

如果基础模型能训练，再逐步启用超分辨率。

---

### ❌ 模块未找到 (Module not found)

**错误信息**:
```
ModuleNotFoundError: No module named 'xxx'
```

**解决方案**:

#### 安装依赖

```bash
pip install -r requirements.txt
```

#### 常见缺失模块

```bash
# PyTorch
pip install torch torchvision

# 其他依赖
pip install wandb tqdm pyyaml pillow
```

---

### ❌ wandb 初始化失败

**错误信息**:
```
wandb: ERROR Unable to authenticate
```

**解决方案**:

#### 登录 wandb

```bash
wandb login
# 输入你的 API key
```

#### 或使用离线模式

```python
# 在 train.py 中修改
wandb.init(
    project="graf250520",
    mode="offline"  # 添加这一行
)
```

---

### ❌ CCSR-ESRGAN 没有效果

**症状**:
- 日志显示使用了 CCSR-ESRGAN
- 但图像质量没有改善

**诊断步骤**:

#### 1. 检查配置

```bash
# 确认配置文件
cat configs/default.yaml | grep -A 5 "ccsr_esrgan:"
```

应该显示：
```yaml
ccsr_esrgan:
  enabled: true
  pretrained_path: 'pretrained_models/RRDB_ESRGAN_x4.pth'
```

#### 2. 检查训练日志

训练开始时应该看到：
```
使用 CCSR-ESRGAN 混合模型
成功加载 XX 个预训练 RRDB 权重
```

#### 3. 检查 wandb 日志

应该有 `loss/sr_consistency` 项

#### 4. 检查模型参数

```python
# 在 Python 中检查
from graf.config import build_models
config = load_config('configs/default.yaml')
generator, _ = build_models(config)

# 检查是否有 ccsr_esrgan
print(hasattr(generator, 'ccsr_esrgan'))  # 应该是 True
```

---

## 📋 快速检查清单

运行训练前，检查以下项目：

- [ ] 已下载预训练模型 `pretrained_models/RRDB_ESRGAN_x4.pth`
- [ ] 已清除 Python 缓存 `find . -name "*.pyc" -delete`
- [ ] 配置文件设置正确 `ccsr_esrgan.enabled: true`
- [ ] 有足够的显存（至少 8GB for batch_size=8）
- [ ] wandb 已登录或设置为离线模式
- [ ] 数据路径正确 `data/RS307_n` 存在
- [ ] 在项目根目录运行 `pwd` 确认

---

## 🆘 获取帮助

如果以上方法都无效：

1. **检查完整错误信息**
   ```bash
   python train.py --config configs/default.yaml 2>&1 | tee error.log
   ```

2. **查看项目文档**
   - ESRGAN_SETUP.md - ESRGAN 使用指南
   - CCSR_ESRGAN_GUIDE.md - CCSR-ESRGAN 详细说明

3. **提供调试信息**
   - Python 版本: `python --version`
   - PyTorch 版本: `python -c "import torch; print(torch.__version__)"`
   - CUDA 版本: `nvidia-smi`
   - 完整错误日志

---

## 📝 常用调试命令

```bash
# 清除所有缓存
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete

# 检查 GPU 状态
nvidia-smi

# 测试导入
python -c "from graf.models.esrgan_model import RRDB; print('OK')"

# 检查配置
python -c "from graf.config import load_config; print(load_config('configs/default.yaml'))"

# 验证数据路径
ls -la data/RS307_n/

# 检查 wandb 状态
wandb status
```

---

祝你训练顺利！ 🚀
