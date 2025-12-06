# NerfAcc 速度问题诊断报告

生成时间: 2025-12-06

## 🚨 关键发现

### 主要问题：数据目录不存在！

在检查代码时发现了一个**严重问题**：

**配置文件中指定的数据目录 `data/RS307_n/` 不存在！**

```bash
$ ls data/RS307_n/
# 输出: Directory not found
```

这可能是导致训练质量差的**根本原因**，而不是 NerfAcc 的问题。

## 问题诊断时间线

### 1. 最初报告的问题
- **症状**: NerfAcc 没有加速训练
- **发现**: `alpha_thre=0.0` 导致无法跳过任何区域
- **修复**: 改为 `alpha_thre=0.01`，后来调整为 `0.001`

### 2. 后续问题
- **症状**: 训练崩溃，生成黑色图像
- **发现**: 采样点减少 97.4%（过于激进）
- **修复**: 降低 `alpha_thre` 到 0.001，暂时禁用 NerfAcc

### 3. 更深层问题
- **症状**: 即使不使用 NerfAcc 和 CCSR，训练结果仍然很差
- **用户反馈**: 数据是用 Blender 生成的
- **关键发现**: **数据目录根本不存在**

## 数据问题分析

### 预期的数据结构

根据代码分析（`graf/datasets.py`），GRAF 预期的数据格式是：

```
data/RS307_n/
├── 0_0.jpg          # category 0, angle 0°
├── 0_1.jpg          # category 0, angle 1°
...
├── 0_359.jpg        # category 0, angle 359°
├── 0.5_0.jpg        # category 1, angle 0°
...
├── 1.5_359.jpg      # category 3, angle 359°
```

**文件命名规则**: `{category_prefix}_{angle_index}.jpg`

**Category 映射**:
- `0_` → v=0.5 (俯仰角 ~60°)
- `0.5_` → v=0.4 (俯仰角 ~66°)
- `1_` → v=0.3 (俯仰角 ~73°)
- `1.5_` → v=0.2 (俯仰角 ~78°)

### 可能的情况

1. **数据在其他位置**
   - 实际训练在不同的机器上
   - 数据在不同的路径
   - 需要确认实际的数据位置

2. **数据尚未生成**
   - Blender 设置已完成，但未导出
   - 需要运行 Blender 脚本导出图像

3. **数据格式不匹配**
   - Blender 生成的数据格式与 GRAF 预期不同
   - 文件命名不符合要求
   - 需要转换脚本

## Blender 相机参数要求

根据 `configs/default.yaml` 和用户提供的 Blender 截图：

| 参数 | 配置值 | Blender 设置 | 状态 |
|------|--------|--------------|------|
| FOV | 20° | 20° | ✓ 匹配 |
| Lens Type | Perspective | Perspective | ✓ 匹配 |
| Radius | 2.5m | 2.5m 半球 | ✓ 匹配 |
| Image Size | 256×256 | ? | 待确认 |
| Near | 1.5 | - | 待确认 |
| Far | 4.5 | - | 待确认 |

**相机位置**: 在半径 2.5m 的半球面上，围绕物体（桥柱/废墟）

## 已修复的 NerfAcc 问题

即使数据问题解决后，以下 NerfAcc 修复仍然重要：

### 1. Alpha Threshold 修复
**文件**: `submodules/nerf_pytorch/run_nerf_mod.py:259`

```python
# 之前: alpha_thre=0.0 (完全不跳过)
# 之后: alpha_thre=0.001 (适度跳过透明区域)
```

### 2. Viewdir 维度修复
**文件**: `submodules/nerf_pytorch/run_nerf_mod.py:186-238`

修复了 occupancy grid 更新时的 viewdir 维度错误。

### 3. 初始化修复
**文件**: `submodules/nerf_pytorch/run_nerf_mod.py:265-267`

确保 `rgb_map`, `acc_map`, `depth_map` 在所有分支都正确初始化。

### 4. GAN 平衡调整
**文件**: `configs/default.yaml`

```yaml
lr_d: 0.00002    # 降低 Discriminator 学习率
lr_g: 0.0008     # 提高 Generator 学习率
reg_param: 20.0  # 降低梯度惩罚
raw_noise_std: 0.1  # 降低 NeRF 噪声
```

### 5. 临时禁用
```yaml
ccsr:
  enabled: False  # 先让 GRAF 正常工作
nerfacc:
  use_nerfacc: false  # 调试阶段禁用
```

## 创建的诊断工具

为帮助调试，创建了以下工具：

### 1. `check_blender_data.py`
**用途**: 验证 Blender 数据格式
```bash
python check_blender_data.py --data_dir data/RS307_n
```

**检查项**:
- ✓ 目录是否存在
- ✓ 文件命名格式
- ✓ 视角覆盖范围
- ✓ 图像尺寸一致性
- ✓ 完整度统计

### 2. `check_data.py`
**用途**: 验证数据加载和统计
```bash
python check_data.py --config configs/default.yaml
```

**检查项**:
- ✓ 数据加载器是否正常
- ✓ Batch 读取
- ✓ 图像统计（均值、方差、范围）
- ✓ Label 正确性
- ✓ 保存样本图像

### 3. `check_camera_params.py`
**用途**: 检查 Blender 相机参数
```bash
python check_camera_params.py
```

**检查项**:
- ✓ 是否存在 transforms.json
- ✓ FOV 是否匹配
- ✓ 相机参数格式
- ✓ 坐标系统

### 4. `debug_nerfacc_performance.py`
**用途**: 测试 NerfAcc 性能
```bash
python debug_nerfacc_performance.py --config configs/default.yaml
```

**测试项**:
- ✓ NerfAcc vs 原始渲染速度
- ✓ 采样点减少统计
- ✓ Occupancy grid 更新时间

### 5. `BLENDER_DATA_FORMAT.md`
**用途**: Blender 数据格式完整指南

包含：
- 详细的目录结构说明
- 文件命名规则
- Blender Python 脚本示例
- 相机参数计算
- 坐标系转换

## 下一步行动计划

### 立即行动（关键）

1. **确认数据位置**
   - 检查实际的数据目录在哪里
   - 或者确认数据是否已从 Blender 导出

2. **验证数据格式**
   ```bash
   python check_blender_data.py
   ```

3. **如果数据不存在，从 Blender 导出**
   - 使用 `BLENDER_DATA_FORMAT.md` 中的脚本
   - 或手动导出并正确命名

4. **测试数据加载**
   ```bash
   python check_data.py
   ```

### 测试训练（数据就绪后）

5. **使用简化配置测试**
   ```bash
   python train.py --config configs/test_basic_config.yaml
   ```

   这将使用：
   - 较小的网络（depth=4, width=128）
   - 低分辨率（64×64）
   - 无 NerfAcc
   - 无 CCSR
   - 快速迭代验证数据是否正确

6. **如果测试成功，逐步启用功能**
   - 先用完整的 GRAF（无 NerfAcc, 无 CCSR）
   - 训练稳定后，启用 NerfAcc
   - 最后启用 CCSR

### 长期优化

7. **创建简单测试场景**
   - 在 Blender 中创建简单场景（单色立方体或球体）
   - 验证整个 pipeline
   - 排除场景复杂度的影响

8. **验证坐标系统**
   - 确认 Blender 导出的相机参数正确
   - 检查是否需要坐标转换
   - 测试渲染的 view 是否匹配

## 问题优先级

### P0 - 阻塞性问题
- [ ] **数据目录不存在** ← 当前最紧急

### P1 - 训练质量问题
- [x] GAN 不平衡（已修复）
- [x] NeRF 噪声过大（已修复）
- [ ] Blender 数据格式验证（待确认）
- [ ] 相机参数匹配（待确认）

### P2 - 性能优化问题
- [x] NerfAcc alpha_thre（已修复）
- [x] NerfAcc viewdir 维度（已修复）
- [ ] NerfAcc 加速验证（数据就绪后测试）

### P3 - 增强功能
- [ ] CCSR 集成（暂时禁用）
- [ ] 多分辨率训练
- [ ] 更复杂的场景

## 结论

**当前最紧迫的问题不是 NerfAcc 的代码，而是数据！**

建议立即：
1. ✓ 确认数据实际位置
2. ✓ 使用 `check_blender_data.py` 验证数据
3. ✓ 如需要，使用 `BLENDER_DATA_FORMAT.md` 中的指南从 Blender 导出数据
4. ✓ 使用 `check_data.py` 测试数据加载
5. ✓ 使用简化配置测试训练

所有的代码修复（NerfAcc, GAN 平衡等）都已完成并提交，但**如果没有正确的数据，训练永远不会成功**。

---

**相关文件**:
- 诊断工具: `check_blender_data.py`, `check_data.py`, `check_camera_params.py`
- 文档: `BLENDER_DATA_FORMAT.md`
- 配置: `configs/default.yaml`, `configs/test_basic_config.yaml`
- 修复提交: 见 git log
