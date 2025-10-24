#!/usr/bin/env python3
"""
設備診斷腳本
用於檢查訓練過程中所有張量的設備位置
"""

import torch
import sys

def check_devices():
    """檢查設備設置"""
    print("=" * 60)
    print("設備診斷報告")
    print("=" * 60)

    # 檢查 CUDA 可用性
    print(f"\n1. CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA 設備數量: {torch.cuda.device_count()}")
        print(f"   當前 CUDA 設備: {torch.cuda.current_device()}")
        print(f"   設備名稱: {torch.cuda.get_device_name(0)}")

    # 設置設備
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n2. 使用的設備: {device}")

    # 模擬訓練循環中的張量創建
    print("\n3. 模擬張量操作:")
    try:
        # 模擬從 DataLoader 獲取數據（默認在 CPU）
        batch_size = 8
        label_cpu = torch.randn(batch_size, 3)
        print(f"   原始 label 設備: {label_cpu.device}")

        # 移到 GPU
        label_gpu = label_cpu.to(device, non_blocking=True)
        print(f"   移動後 label 設備: {label_gpu.device}")

        # 提取 first_label
        first_label = label_gpu[:,0].long()
        print(f"   first_label 設備: {first_label.device}")

        # 創建 one_hot
        one_hot = torch.zeros(batch_size, 1, device=device)
        print(f"   one_hot 設備: {one_hot.device}")

        # 測試 scatter 操作
        one_hot.scatter_(1, first_label.unsqueeze(1), 1)
        print(f"   ✅ scatter_ 操作成功!")

        print("\n✅ 所有設備操作正常!")
        return True

    except RuntimeError as e:
        print(f"\n❌ 錯誤: {e}")
        print("\n可能的原因:")
        print("1. label 未正確移到 GPU")
        print("2. first_label 和 one_hot 在不同設備")
        return False

def test_dataloader_device():
    """測試 DataLoader 的設備行為"""
    print("\n" + "=" * 60)
    print("DataLoader 設備測試")
    print("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 創建簡單的數據集
    dummy_data = torch.randn(16, 3, 64, 64)
    dummy_labels = torch.randint(0, 10, (16, 3))

    dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        pin_memory=True
    )

    print(f"\n從 DataLoader 提取第一個 batch:")
    for x, label in dataloader:
        print(f"   x 設備: {x.device}")
        print(f"   label 設備: {label.device}")

        # 測試移動到 GPU
        x = x.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        print(f"\n   移動後 x 設備: {x.device}")
        print(f"   移動後 label 設備: {label.device}")

        # 測試操作
        first_label = label[:,0].long()
        print(f"   first_label 設備: {first_label.device}")

        one_hot = torch.zeros(label.size(0), 1, device=device)
        print(f"   one_hot 設備: {one_hot.device}")

        try:
            one_hot.scatter_(1, first_label.unsqueeze(1), 1)
            print(f"\n   ✅ DataLoader 測試成功!")
        except RuntimeError as e:
            print(f"\n   ❌ DataLoader 測試失敗: {e}")

        break  # 只測試第一個 batch

def print_recommendations():
    """打印建議"""
    print("\n" + "=" * 60)
    print("修復建議")
    print("=" * 60)
    print("""
如果測試失敗，請檢查以下幾點：

1. 確保你已經從 Git 拉取最新代碼:
   cd /Data/home/vicky/graf250916/
   git pull origin claude/code-review-011CUKo9GJraRmNXfGkcWKxR

2. 檢查 train.py 第 180-181 行是否有以下代碼:
   x_real = x_real.to(device, non_blocking=True)
   label = label.to(device, non_blocking=True)

3. 如果還有問題，手動添加調試代碼:
   在 train.py 第 184 行後添加:

   print(f"Debug - label device: {label.device}")
   print(f"Debug - first_label device: {first_label.device}")
   print(f"Debug - one_hot device: {one_hot.device}")

4. 確保沒有混用不同版本的代碼文件

5. 清理 Python 緩存:
   find . -type d -name "__pycache__" -exec rm -r {} +
   find . -type f -name "*.pyc" -delete
""")

if __name__ == "__main__":
    print("\n🔍 開始設備診斷...\n")

    # 運行測試
    basic_test = check_devices()
    test_dataloader_device()
    print_recommendations()

    # 總結
    print("\n" + "=" * 60)
    if basic_test:
        print("✅ 診斷完成 - 設備配置正常")
        print("\n如果訓練還有問題，請確保:")
        print("1. 使用最新版本的 train.py")
        print("2. 清理 Python 緩存")
        print("3. 重啟 Python 環境")
    else:
        print("❌ 診斷完成 - 發現設備問題")
        print("\n請檢查上面的修復建議")
    print("=" * 60 + "\n")
