# CCSR Discriminator 訓練修復

## 問題描述

### 原始問題
當使用 `x_fake_sr`（經過 CCSR 處理的生成圖片）輸入 Discriminator 時，訓練無法正常進行。而使用 `x_fake_nerf`（NeRF 直接輸出）則可以正常訓練。

### 根本原因

**Discriminator 看到的真實圖片和生成圖片經過了不同的處理流程**：

| 類型 | 原始處理流程 | 結果 |
|------|------------|------|
| 真實圖片 | 原始圖片 → Discriminator | 無 SR artifacts |
| 生成圖片 | NeRF 輸出 → 下採樣 → CCSR/ESRGAN → 上採樣 → CEM → Discriminator | 有 SR artifacts |

這導致 Discriminator 學習到：
- ✗ "有 SR artifacts" = 假圖片
- ✗ "無 SR artifacts" = 真圖片

**這是錯誤的學習目標！** Discriminator 應該學習區分「內容的真實性」，而不是「是否有 SR 處理痕跡」。

## 解決方案

### SUPER-NERF 架構

根據 SUPER-NERF 論文的做法，**真實圖片和生成圖片都應該通過相同的 SR 處理流程**：

| 類型 | 修復後的處理流程 |
|------|----------------|
| 真實圖片 | 原始圖片 → 下採樣 → CCSR/ESRGAN → 上採樣 → CEM → Discriminator |
| 生成圖片 | NeRF 輸出 → 下採樣 → CCSR/ESRGAN → 上採樣 → CEM → Discriminator |

這樣 Discriminator 就能專注於學習：
- ✓ "生成的內容" vs "真實的內容"
- ✓ 而不是 "SR artifacts" vs "原始紋理"

## 實現細節

### 修改位置：`train.py` Line 186-246

#### 1. 真實圖片也通過 CCSR 處理

```python
# 如果使用 SR，真實圖片也要通過 CCSR 處理
if use_sr:
    # 將 patch 格式轉換為圖像格式
    total_elements = rgbs.numel()
    rgbs_reshaped = rgbs.view(batch_size, total_elements // (batch_size * 3), 3)
    patch_size = int(np.sqrt(rgbs_reshaped.shape[1]))
    real_images = rgbs_reshaped.view(batch_size, patch_size, patch_size, 3).permute(0, 3, 1, 2)

    # 下採樣到低分辨率
    lr_size = max(8, patch_size // 4)
    lr_real_images = F.interpolate(real_images, size=(lr_size, lr_size),
                                  mode='bilinear', align_corners=False)

    # 對每個樣本應用 CCSR（根據視角）
    sr_real_results = []
    for i in range(batch_size):
        angle_idx = int(label[i, 2].item())
        view_idx = (angle_idx * 8) // 360  # 映射到 0-7

        if generator.use_ccsr_esrgan:
            result = generator.ccsr_esrgan(lr_real_images[i:i+1], view_idx)
        elif generator.use_esrgan:
            result = generator.esrgan(lr_real_images[i:i+1])
        elif generator.use_ccsr:
            result = generator.ccsr(lr_real_images[i:i+1], view_idx)

        sr_real_results.append(result)

    sr_real_combined = torch.cat(sr_real_results, dim=0)
    sr_real_resized = F.interpolate(sr_real_combined, size=(patch_size, patch_size),
                                   mode='bilinear', align_corners=False)

    # 轉換回 patch 格式
    rgbs_sr = sr_real_resized.permute(0, 2, 3, 1).contiguous().view(-1, 3)
    rgbs_sr.requires_grad_(True)

    # 使用 SR 處理後的真實圖片
    d_real, label_real = discriminator(rgbs_sr, label)
else:
    rgbs.requires_grad_(True)
    d_real, label_real = discriminator(rgbs, label)
```

#### 2. 梯度懲罰也使用正確的變數

```python
# 使用正確的 rgbs 變數計算梯度懲罰
rgbs_for_reg = rgbs_sr if use_sr else rgbs
reg = 80. * compute_grad2(d_real, rgbs_for_reg).mean()
```

## CCSR 架構回顧

### CCLC (Consistency-Controlling Latent Code)
- 為每個視角學習獨特的潛在代碼
- 確保不同視角之間的一致性
- 實現於 `graf/models/ccsr.py:8-25`

### CEM (Consistency-Enforcing Module)
- 確保 SR 輸出與原始 LR 輸入一致
- 通過下採樣 SR 輸出並計算與 LR 的殘差
- 將殘差上採樣並加回 SR 輸出進行修正
- 實現於 `graf/models/ccsr.py:28-59`

### 處理流程

```
LR 圖片 (16x16)
    ↓
上採樣 + concat CCLC
    ↓
2D SR Module (ESRGAN/簡化網路)
    ↓
SR 輸出 (64x64)
    ↓
CEM：
  1. 下採樣 SR 到 LR 尺寸
  2. 計算殘差：LR - 下採樣SR
  3. 上採樣殘差到 SR 尺寸
  4. 修正：SR + α * 殘差
    ↓
最終 SR 輸出 (64x64)
```

## 預期效果

### 訓練穩定性
- ✅ Discriminator 能夠學習到有意義的特徵
- ✅ 訓練損失曲線更穩定
- ✅ 不會出現 mode collapse

### 生成品質
- ✅ SR 輸出具有更好的視覺品質
- ✅ 多視角一致性更好（透過 CCLC）
- ✅ 與原始 NeRF 輸出更一致（透過 CEM）

## 測試方法

### 1. 檢查訓練 log
```bash
# 觀察 Discriminator 損失
# D_real 和 D_fake 應該在合理範圍內（例如 0.3-0.7）
# 不應該出現 D_fake → 0 或 D_fake → 1 的情況
```

### 2. 視覺化檢查
在訓練過程中保存以下圖片比較：
- 原始真實圖片
- SR 處理後的真實圖片
- NeRF 輸出
- SR 處理後的 NeRF 輸出

### 3. 量化指標
- FID (Fréchet Inception Distance)
- LPIPS (Learned Perceptual Image Patch Similarity)
- SSIM (Structural Similarity Index)

## 相關文件

- `graf/models/ccsr.py`: CCSR 模組實現
- `graf/models/generator.py`: Generator 中的 SR 處理
- `CCSR_ESRGAN_GUIDE.md`: CCSR-ESRGAN 詳細指南

## 參考

- SUPER-NERF 論文
- CCSR (Consistency-Controlled Super-Resolution)
- ESRGAN (Enhanced Super-Resolution GAN)
