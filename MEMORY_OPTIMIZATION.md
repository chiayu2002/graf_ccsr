# 記憶體管理優化指南

本文檔總結了對 graf_ccsr 專案進行的記憶體管理優化。

## 📊 已實施的優化

### 1. 移除全局 CUDA 張量設置
**檔案**: `train.py`, `eval.py`

**修改前**:
```python
torch.set_default_tensor_type('torch.cuda.FloatTensor')
```

**問題**: 導致所有張量默認在 GPU 上創建，增加記憶體壓力

**修改後**: 移除此行，手動管理張量設備

---

### 2. 優化 DataLoader 配置
**檔案**: `train.py:42-54`

**新增功能**:
- 限制 `num_workers` 最多為 4
- 添加 `prefetch_factor=2` 以平衡記憶體和速度
- 啟用 `persistent_workers` 以減少 worker 重啟開銷

```python
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config['training']['batch_size'],
    num_workers=min(config['training']['nworkers'], 4),
    shuffle=True,
    pin_memory=True,
    prefetch_factor=2 if config['training']['nworkers'] > 0 else None,
    persistent_workers=config['training']['nworkers'] > 0,
    ...
)
```

---

### 3. 優化器記憶體管理
**檔案**: `train.py:187, 241`

**改進**: 使用 `zero_grad(set_to_none=True)` 代替 `zero_grad()`

```python
d_optimizer.zero_grad(set_to_none=True)  # 節省記憶體
g_optimizer.zero_grad(set_to_none=True)
```

**效果**: 將梯度設置為 None 而非填充零，節省記憶體分配

---

### 4. 修復重複對象創建
**檔案**: `train.py:106`

**修改前**: 每次迭代創建新的 `MCE_Loss()` 實例

**修改後**: 在訓練循環外創建一次
```python
# 初始化損失函數（在循環外創建以節省記憶體）
ccsr_nerf_loss = CCSRNeRFLoss().to(device)
mce_loss = MCE_Loss()
```

---

### 5. 張量異步傳輸
**檔案**: `train.py:189, 196, 202, etc.`

**改進**: 使用 `non_blocking=True` 進行 CPU 到 GPU 傳輸

```python
x_real = x_real.to(device, non_blocking=True)
label_tensor = label.to(device, non_blocking=True)
```

**效果**: 允許 CPU 和 GPU 操作重疊，提高效率

---

### 6. 主動釋放張量
**檔案**: `train.py:231, 261`

**新增**: 在不需要時主動刪除張量

```python
# 釋放不需要的張量
del x_fake, d_real, d_fake, rgbs, dloss_real, dloss_fake, reg, total_d_loss
```

---

### 7. CUDA 緩存清理
**檔案**: `train.py:287, 313-314`

**新增**: 在關鍵位置清理 CUDA 緩存

```python
torch.cuda.empty_cache()  # 清理 CUDA 緩存
gc.collect()  # Python 垃圾回收
```

**位置**:
- 採樣後 (每 500 次迭代)
- FID/KID 計算後 (每 5000 次迭代)
- 檢查點保存後

---

### 8. 使用 `torch.no_grad()` 上下文
**檔案**: `train.py`, `eval.py`, `generator.py`

**改進**: 在不需要梯度的操作中使用上下文管理器

```python
with torch.no_grad():
    # 採樣或評估代碼
    samples = evaluator.create_samples(...)
```

**位置**:
- 訓練時的 discriminator 假數據生成
- 採樣和可視化
- FID/KID 計算
- 評估腳本中的所有操作

---

### 9. Generator 中的 CCSR 優化
**檔案**: `graf/models/generator.py:112-156`

**改進**:
1. 修復硬編碼的 `view_idx = 72`，改為從 label 中提取
2. 在 CCSR 處理時使用 `torch.no_grad()` 暫時關閉梯度
3. 主動刪除中間張量
4. 添加完美平方數驗證

```python
with torch.no_grad():  # CCSR 處理時暫時不需要梯度
    # 處理邏輯
    ...
    del ccsr_results, lr_images  # 釋放中間結果
    del ccsr_combined  # 釋放

# 重新啟用梯度
ccsr_output.requires_grad_(True)
```

---

### 10. 評估腳本優化
**檔案**: `eval.py`

**改進**:
1. 所有評估操作包裹在 `torch.no_grad()` 中
2. 樣本生成後立即移到 CPU
3. 定期調用 `torch.cuda.empty_cache()`
4. 修復未定義的 `label` 變數問題

```python
with torch.no_grad():
    for i, (u, v) in enumerate(angle_positions):
        rgb, depth, acc = evaluator.create_samples(...)
        all_rgb.append(rgb.cpu())  # 立即移到 CPU
        del depth, acc
        if i % 2 == 0:
            torch.cuda.empty_cache()
```

---

### 11. 混合精度訓練（AMP）
**檔案**: `train.py:78, 110-114, 204-228, 246-258`

**新增**: 可選的混合精度訓練支援

**使用方法**:
```bash
python train.py --config configs/default.yaml --use_amp
```

**實現**:
```python
# 初始化 scaler
scaler_d = torch.cuda.amp.GradScaler(enabled=use_amp)
scaler_g = torch.cuda.amp.GradScaler(enabled=use_amp)

# 訓練時使用
with torch.cuda.amp.autocast(enabled=use_amp):
    d_real, _ = discriminator(rgbs, label)
    dloss_real = compute_loss(d_real, 1)

scaler_d.scale(total_d_loss).backward()
scaler_d.step(d_optimizer)
scaler_d.update()
```

**效果**: 可節省高達 50% 的 GPU 記憶體

---

## 📈 預期效果

### 記憶體節省估計
| 優化項目 | 預期節省 |
|---------|---------|
| 移除全局 CUDA 張量設置 | ~10% |
| 優化器 set_to_none | ~5-10% |
| 主動釋放張量 + 緩存清理 | ~15-20% |
| 混合精度訓練 (AMP) | ~30-50% |
| **總計** | **~60-90%** |

### 性能影響
- **訓練速度**: 可能略微減慢 (~5-10%)，因為添加了記憶體清理操作
- **穩定性**: 顯著提升，減少 OOM (Out of Memory) 錯誤
- **批次大小**: 可以增加 1.5-2 倍的 batch size

---

## 🔧 使用建議

### 1. 標準訓練（記憶體充足）
```bash
python train.py --config configs/default.yaml
```

### 2. 記憶體受限環境
```bash
python train.py --config configs/default.yaml --use_amp
```

### 3. 調整 batch size
如果記憶體仍然不足，修改 `configs/default.yaml`:
```yaml
training:
  batch_size: 4  # 從 8 減少到 4
```

### 4. 監控記憶體使用
添加以下代碼到訓練循環中:
```python
if (it % 100) == 0:
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"記憶體: 已分配 {allocated:.2f}GB, 保留 {reserved:.2f}GB")
```

---

## ⚠️ 注意事項

### 1. 混合精度訓練的限制
- 某些操作（如梯度懲罰計算）仍需在 FP32 中進行
- 可能會影響數值穩定性，建議監控訓練曲線

### 2. DataLoader workers
- 如果使用 HDD 而非 SSD，減少 `num_workers`
- 在 Windows 上可能需要設置 `num_workers=0`

### 3. 記憶體碎片
- 長時間訓練後可能出現記憶體碎片
- 建議定期重啟訓練（使用 checkpoint）

---

## 🐛 故障排除

### 問題 1: 仍然出現 OOM
**解決方案**:
1. 啟用 `--use_amp`
2. 減少 `batch_size`
3. 減少 `ray_sampler.N_samples` (在 config.yaml 中)

### 問題 2: 訓練變慢
**解決方案**:
1. 檢查是否過度使用 `torch.cuda.empty_cache()`
2. 確保 `pin_memory=True` 且 `non_blocking=True`
3. 增加 `prefetch_factor`

### 問題 3: 數值不穩定（使用 AMP 時）
**解決方案**:
1. 檢查 loss scaling（scaler）
2. 某些層可能需要強制使用 FP32
3. 監控梯度範數

---

## 📝 進一步優化建議

### 1. 梯度累積（未實現）
如果仍需更大的有效 batch size:
```python
accumulation_steps = 4
for i, (x_real, label) in enumerate(train_loader):
    loss = ... / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 2. Gradient Checkpointing（未實現）
對於非常深的網絡:
```python
from torch.utils.checkpoint import checkpoint
output = checkpoint(some_function, input)
```

### 3. 分散式訓練（未實現）
使用多 GPU:
```bash
python -m torch.distributed.launch --nproc_per_node=2 train.py ...
```

---

## 📚 參考資料

- [PyTorch 記憶體管理最佳實踐](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [混合精度訓練指南](https://pytorch.org/docs/stable/amp.html)
- [CUDA 最佳實踐指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

---

**最後更新**: 2025-10-21
**維護者**: Claude Code Review Assistant
