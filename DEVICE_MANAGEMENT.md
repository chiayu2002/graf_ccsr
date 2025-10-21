# GPU 設備管理檢查清單

本文檔列出了所有已修復的設備不匹配問題，以及如何避免未來出現類似錯誤。

## 🔴 已修復的設備錯誤

### 錯誤 1: DataLoader 中的數據未移到 GPU
**位置**: `train.py:176-183`

**錯誤訊息**:
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

**原因**:
```python
# 錯誤的代碼
for x_real, label in tqdm(train_loader):
    first_label = label[:,0].long()  # label 在 CPU 上
    one_hot = torch.zeros(batch_size, 1, device=device)  # one_hot 在 GPU 上
    one_hot.scatter_(1, first_label.unsqueeze(1), 1)  # ❌ 設備不匹配！
```

**修復**:
```python
# 正確的代碼
for x_real, label in tqdm(train_loader):
    # 立即將數據移到 GPU
    x_real = x_real.to(device, non_blocking=True)
    label = label.to(device, non_blocking=True)

    first_label = label[:,0].long()  # ✅ 現在在 GPU 上
    one_hot = torch.zeros(batch_size, 1, device=device)
    one_hot.scatter_(1, first_label.unsqueeze(1), 1)  # ✅ 都在 GPU 上
```

**影響文件**: `train.py:180-181`

---

### 錯誤 2: 重複的設備轉換
**位置**: `train.py:205, 214, 217, 246, 247`

**問題**:
```python
# 效率低下且容易出錯
label.to(device, non_blocking=True)  # 轉換多次
label.to(device, non_blocking=True)  # 重複轉換
label.to(device, non_blocking=True)  # 又一次轉換
```

**修復**:
```python
# 在循環開始時轉換一次
label = label.to(device, non_blocking=True)
# 之後直接使用 label
```

**影響**: 減少不必要的設備轉換，提高效率

---

### 錯誤 3: 採樣時創建的張量未指定設備
**位置**: `train.py:291`

**錯誤代碼**:
```python
label_test = torch.tensor([[0] if i < 4 else [0] for i in range(batch_size)])
# ❌ 默認在 CPU 上
```

**修復**:
```python
label_test = torch.tensor([[0] if i < 4 else [0] for i in range(batch_size)], device=device)
# ✅ 明確指定設備
```

---

### 錯誤 4: eval.py 中的 create_labels 函數
**位置**: `eval.py:122`

**錯誤代碼**:
```python
def create_labels(num_samples, label_value):
    return torch.full((num_samples, 1), label_value)  # ❌ 在 CPU 上
```

**修復**:
```python
def create_labels(num_samples, label_value):
    return torch.full((num_samples, 1), label_value, device=device)  # ✅ 在 GPU 上
```

---

## ✅ 設備管理最佳實踐

### 1. 在訓練循環開始時立即移動數據

```python
# ✅ 好的做法
for x, label in dataloader:
    x = x.to(device, non_blocking=True)
    label = label.to(device, non_blocking=True)
    # 後續所有操作使用已移動的張量

# ❌ 不好的做法
for x, label in dataloader:
    output = model(x.to(device))  # 每次都轉換
    loss = criterion(output, label.to(device))  # 重複轉換
```

### 2. 創建張量時明確指定設備

```python
# ✅ 好的做法
zeros = torch.zeros(size, device=device)
ones = torch.ones(size, device=device)
tensor = torch.tensor(data, device=device)
full = torch.full(size, value, device=device)

# ❌ 不好的做法
zeros = torch.zeros(size).to(device)  # 先在 CPU 創建再移動
```

### 3. 使用 non_blocking=True 加速異步傳輸

```python
# ✅ 異步傳輸（當 pin_memory=True 時更快）
x = x.to(device, non_blocking=True)

# ❌ 同步傳輸（較慢）
x = x.to(device)
```

### 4. 在推理時使用 torch.no_grad()

```python
# ✅ 節省記憶體
with torch.no_grad():
    output = model(x)

# ❌ 會追蹤梯度，浪費記憶體
output = model(x)
```

---

## 🔍 如何檢測設備不匹配問題

### 方法 1: 在關鍵位置添加斷言

```python
def forward(self, x, label):
    # 檢查輸入是否在同一設備
    assert x.device == label.device, f"Device mismatch: x on {x.device}, label on {label.device}"
    assert x.device.type == 'cuda', f"Expected CUDA device, got {x.device}"

    # 繼續處理
    ...
```

### 方法 2: 打印張量設備

```python
print(f"x device: {x.device}")
print(f"label device: {label.device}")
print(f"model device: {next(model.parameters()).device}")
```

### 方法 3: 使用調試工具

```python
# 啟用異常檢測
torch.autograd.set_detect_anomaly(True)
```

---

## 📋 檢查清單

在添加新代碼時，檢查以下項目：

### DataLoader
- [ ] 從 DataLoader 獲取的數據是否立即移到 GPU？
- [ ] 是否使用 `non_blocking=True`？
- [ ] 是否在 DataLoader 中設置 `pin_memory=True`？

### 張量創建
- [ ] 所有 `torch.tensor()` 是否指定 `device=device`？
- [ ] 所有 `torch.zeros()`, `torch.ones()` 是否指定 `device=device`？
- [ ] 所有 `torch.full()`, `torch.empty()` 是否指定 `device=device`？

### 模型操作
- [ ] 模型是否已移到 GPU（`model.to(device)`）？
- [ ] 損失函數是否已移到 GPU（如需要）？
- [ ] 所有輸入張量是否在同一設備上？

### 推理和評估
- [ ] 是否使用 `torch.no_grad()` 包裹推理代碼？
- [ ] 評估數據是否正確移到 GPU？
- [ ] 如果需要在 CPU 上處理結果，是否使用 `.cpu()`？

---

## 🛠️ 常見錯誤模式和修復

### 錯誤模式 1: 部分張量在 GPU，部分在 CPU

```python
# ❌ 錯誤
x = x.to(device)
y = torch.zeros(10)  # 在 CPU 上
z = x + y  # RuntimeError!

# ✅ 修復
x = x.to(device)
y = torch.zeros(10, device=device)  # 在 GPU 上
z = x + y  # OK
```

### 錯誤模式 2: 循環中重複轉換

```python
# ❌ 錯誤（效率低）
for batch in dataloader:
    x = batch['x'].to(device)
    y = batch['y'].to(device)
    z = batch['z'].to(device)

# ✅ 修復
for batch in dataloader:
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
```

### 錯誤模式 3: 忘記將標籤移到 GPU

```python
# ❌ 錯誤
for images, labels in dataloader:
    images = images.to(device)
    outputs = model(images)
    loss = criterion(outputs, labels)  # labels 還在 CPU！

# ✅ 修復
for images, labels in dataloader:
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    outputs = model(images)
    loss = criterion(outputs, labels)
```

---

## 📊 修復總結

| 文件 | 行數 | 問題 | 修復 |
|-----|------|------|------|
| train.py | 180-181 | 數據未移到 GPU | 添加 `.to(device)` |
| train.py | 205, 214, 217 | 重複轉換 | 移除重複調用 |
| train.py | 246-247 | 重複轉換 | 移除重複調用 |
| train.py | 291 | label_test 在 CPU | 添加 `device=device` |
| eval.py | 122 | create_labels 在 CPU | 添加 `device=device` |

---

## 🔬 測試建議

在修改後，建議運行以下測試：

```bash
# 1. 快速測試（運行幾個迭代）
python train.py --config configs/default.yaml

# 2. 檢查設備使用
# 在代碼中添加：
if it == 0:
    print(f"x_real device: {x_real.device}")
    print(f"label device: {label.device}")
    print(f"Generator device: {next(generator.parameters()).device}")
    print(f"Discriminator device: {next(discriminator.parameters()).device}")

# 3. 記憶體監控
watch -n 1 nvidia-smi
```

---

## 📚 參考資料

- [PyTorch 設備管理文檔](https://pytorch.org/docs/stable/notes/cuda.html)
- [PyTorch 性能優化指南](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [CUDA 最佳實踐](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

---

**最後更新**: 2025-10-21
**維護者**: Claude Code Review Assistant

---

## ⚡ 快速參考

### 常用設備操作

```python
# 檢查設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 移動張量
tensor = tensor.to(device)
tensor = tensor.cuda()  # 等同於 .to('cuda')
tensor = tensor.cpu()   # 移回 CPU

# 創建時指定設備
tensor = torch.zeros(10, device=device)

# 檢查張量設備
print(tensor.device)
assert tensor.is_cuda

# 異步傳輸
tensor = tensor.to(device, non_blocking=True)
```

### 調試命令

```python
# 打印所有張量的設備
for name, param in model.named_parameters():
    print(f"{name}: {param.device}")

# 檢查計算圖中的設備
torch.autograd.set_detect_anomaly(True)
```
