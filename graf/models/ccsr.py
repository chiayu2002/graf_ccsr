import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from graf.models.esrgan_model import RRDB


class ConsistencyControllingLatentCode(nn.Module):
    """一致性控制潛在代碼 (CCLC)"""
    
    def __init__(self, num_views: int, lr_height: int, lr_width: int, scale_factor: int = 4):
        super().__init__()
        self.num_views = num_views
        self.scale_factor = scale_factor
        
        # 為每個視角初始化可學習的潛在代碼
        self.latent_codes = nn.Parameter(
            torch.randn(num_views, 3, scale_factor * lr_height, scale_factor * lr_width) * 0.01
        )
        
    def forward(self, view_idx: int) -> torch.Tensor:
        """獲取特定視角的潛在代碼"""
        if isinstance(view_idx, torch.Tensor):
            view_idx = view_idx.item()
        return self.latent_codes[view_idx % self.num_views]


class ConsistencyEnforcingModule(nn.Module):
    """一致性執行模組 (CEM)"""
    
    def __init__(self, blur_kernel_size: int = 3):
        super().__init__()
        # 定義模糊核
        self.blur_kernel_size = blur_kernel_size
        kernel = torch.ones(1, 1, blur_kernel_size, blur_kernel_size) / (blur_kernel_size * blur_kernel_size)
        self.register_buffer('blur_kernel', kernel)
        
    def forward(self, sr_image: torch.Tensor, lr_image: torch.Tensor) -> torch.Tensor:
        """使用模糊核的CEM實現"""
        # 使用模糊核對SR圖像進行模糊處理
        blurred = F.conv2d(sr_image, 
                        self.blur_kernel.expand(sr_image.size(1), -1, -1, -1), 
                        padding=self.blur_kernel_size//2, 
                        groups=sr_image.size(1))
        downsampled = F.interpolate(blurred, size=lr_image.shape[-2:], mode='bilinear', align_corners=False)
        """執行一致性強制"""
        # 簡化的CEM實現
        # 對SR圖像進行下採樣
        # scale = sr_image.shape[-1] // lr_image.shape[-1]
        # downsampled = F.interpolate(sr_image, size=lr_image.shape[-2:], mode='bilinear', align_corners=False)
        
        # 計算殘差並上採樣
        residual = lr_image - downsampled
        upsampled_residual = F.interpolate(residual, size=sr_image.shape[-2:], mode='bilinear', align_corners=False)
        
        # 應用修正
        refined_sr_image = sr_image + 0.5 * upsampled_residual
        
        return refined_sr_image

# class ConsistencyEnforcingModule(nn.Module):
#     def __init__(self, blur_kernel_size: int = 3, noise_std: float = 0.0):
#         super().__init__()
#         # 創建高斯模糊核
#         self.blur_kernel_size = blur_kernel_size
#         self.noise_std = noise_std
#         self._create_blur_kernel()
        
#     def _create_blur_kernel(self):
#         # 創建高斯核而不是均勻核
#         sigma = self.blur_kernel_size / 3.0
#         kernel_1d = torch.exp(-0.5 * ((torch.arange(self.blur_kernel_size) - self.blur_kernel_size // 2) / sigma) ** 2)
#         kernel_1d = kernel_1d / kernel_1d.sum()
#         kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
#         self.register_buffer('blur_kernel', kernel_2d.unsqueeze(0).unsqueeze(0))
    
#     def forward(self, sr_image: torch.Tensor, lr_image: torch.Tensor) -> torch.Tensor:
#         """基於論文公式的CEM實現"""
#         # H矩阵操作：模糊+下採樣
#         blurred = F.conv2d(sr_image, self.blur_kernel.expand(sr_image.size(1), -1, -1, -1), 
#                           padding=self.blur_kernel_size//2, groups=sr_image.size(1))
#         downsampled = F.interpolate(blurred, size=lr_image.shape[-2:], mode='bilinear')
        
#         # 添加噪聲模型（如果指定）
#         if self.noise_std > 0:
#             noise = torch.randn_like(downsampled) * self.noise_std
#             downsampled = downsampled + noise
        
#         # 計算正交投影
#         # P_N(H)⊥ = H^T(HH^T)^(-1)H
#         residual = lr_image - downsampled
#         upsampled_residual = F.interpolate(residual, size=sr_image.shape[-2:], mode='bilinear')
        
#         # 修正SR圖像
#         refined_sr = sr_image + upsampled_residual
        
#         return refined_sr
    

class CCSR(nn.Module):
    """一致性控制超分辨率模組（簡化版）"""

    def __init__(self, num_views: int, lr_height: int, lr_width: int, scale_factor: int = 4):
        super().__init__()

        self.scale_factor = scale_factor
        self.cclc = ConsistencyControllingLatentCode(num_views, 64, 64, 1)
        self.cem = ConsistencyEnforcingModule()

        # 簡化的超分辨率網絡
        self.sr_network = self._build_sr_network(scale_factor)
        self.activation = nn.LeakyReLU(0.2)

    def _build_sr_network(self, scale_factor: int) -> nn.Module:
        """構建超分辨率網絡"""
        return nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),  # 6 = 3(LR) + 3(latent)
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 3 * scale_factor * scale_factor, 3, padding=1),
            nn.PixelShuffle(scale_factor)
        )

    def forward(self, lr_image: torch.Tensor, view_idx: int) -> torch.Tensor:
        """CCSR前向傳播"""
        batch_size = lr_image.shape[0]

        # 獲取潛在代碼
        latent_code = self.cclc(view_idx)
        latent_code = latent_code.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # 上採樣低分辨率圖像
        lr_upsampled = F.interpolate(lr_image, size=latent_code.shape[-2:], mode='bilinear', align_corners=False)

        # 連接輸入
        combined_input = torch.cat([lr_upsampled, latent_code], dim=1)

        # 生成超分辨率圖像
        sr_output = self.sr_network(combined_input)
        sr_output = self.activation(sr_output)
        sr_output = torch.clamp(sr_output, -1, 1)  # 匹配GRAF的圖像範圍 [-1, 1]

        # 應用CEM
        refined_sr = self.cem(sr_output, lr_image)

        return refined_sr


class CCSR_ESRGAN(nn.Module):
    """
    一致性控制超分辨率模組 - ESRGAN 版本

    結合 CCSR 的多視角一致性機制和 ESRGAN 的強大超分辨率能力：
    - CCLC: 為每個視角學習獨特的潛在代碼
    - ESRGAN RRDB: 強大的超分辨率網絡
    - CEM: 確保 SR 圖像與 LR 圖像的一致性
    """

    def __init__(self, num_views: int, lr_height: int, lr_width: int, scale_factor: int = 4,
                 num_rrdb_blocks: int = 16, nf: int = 64, gc: int = 32,
                 pretrained_path: str = None, freeze_rrdb: bool = False):
        """
        Args:
            num_views: 視角數量
            lr_height: 低分辨率圖像高度（不使用，保留接口兼容性）
            lr_width: 低分辨率圖像寬度（不使用，保留接口兼容性）
            scale_factor: 上採樣倍率
            num_rrdb_blocks: RRDB blocks 數量
            nf: 特徵圖通道數
            gc: RRDB 的 growth channel
            pretrained_path: ESRGAN 預訓練權重路徑
            freeze_rrdb: 是否凍結 RRDB 參數
        """
        super().__init__()

        self.scale_factor = scale_factor
        self.num_views = num_views

        # 1. 多視角潛在代碼 (CCLC)
        self.cclc = ConsistencyControllingLatentCode(num_views, 64, 64, 1)

        # 2. 融合層：將 LR 圖像(3ch) + 潛在代碼(3ch) 融合為特徵
        self.fusion_conv = nn.Conv2d(6, nf, 3, 1, 1, bias=True)

        # 3. RRDB Trunk (ESRGAN 核心)
        self.rrdb_trunk = nn.ModuleList()
        for _ in range(num_rrdb_blocks):
            self.rrdb_trunk.append(RRDB(nf, gc))

        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # 4. 上採樣層 (x4 = 2x upsample twice)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # 5. 最後的高分辨率卷積
        self.hrconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)

        # 6. 一致性強制模組 (CEM)
        self.cem = ConsistencyEnforcingModule()

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # 加載預訓練權重（僅 RRDB 部分）
        if pretrained_path is not None:
            self._load_pretrained_rrdb(pretrained_path)

        # 凍結 RRDB 參數
        if freeze_rrdb:
            self._freeze_rrdb()

    def _load_pretrained_rrdb(self, pretrained_path):
        """從預訓練的 ESRGAN 加載 RRDB trunk 權重"""
        import os
        if not os.path.exists(pretrained_path):
            print(f"警告: 預訓練模型不存在: {pretrained_path}")
            print("CCSR-ESRGAN 將使用隨機初始化")
            return

        try:
            pretrained_dict = torch.load(pretrained_path, map_location='cpu')

            # 處理不同的權重格式
            if 'params' in pretrained_dict:
                pretrained_dict = pretrained_dict['params']
            elif 'model_state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['model_state_dict']
            elif 'state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['state_dict']

            # 只加載 RRDB trunk 的權重
            model_dict = self.state_dict()
            pretrained_rrdb = {}

            print(f"預訓練模型包含 {len(pretrained_dict)} 個權重")

            for k, v in pretrained_dict.items():
                # 移除可能的 'model.' 前綴
                k_clean = k.replace('model.', '')

                # 匹配 RRDB trunk 的權重
                if 'RRDB_trunk' in k_clean:
                    # 將 ESRGAN 的 RRDB_trunk 映射到 CCSR 的 rrdb_trunk
                    new_k = k_clean.replace('RRDB_trunk', 'rrdb_trunk')

                    if new_k in model_dict:
                        if model_dict[new_k].shape == v.shape:
                            pretrained_rrdb[new_k] = v

                # 匹配 trunk_conv
                elif 'trunk_conv' in k_clean:
                    if k_clean in model_dict:
                        if model_dict[k_clean].shape == v.shape:
                            pretrained_rrdb[k_clean] = v

            if pretrained_rrdb:
                model_dict.update(pretrained_rrdb)
                self.load_state_dict(model_dict, strict=False)
                print(f"✓ 成功加載 {len(pretrained_rrdb)} 個預訓練 RRDB 權重")

                # 統計加載的權重
                rrdb_count = sum(1 for k in pretrained_rrdb.keys() if 'rrdb_trunk' in k)
                trunk_conv_count = sum(1 for k in pretrained_rrdb.keys() if 'trunk_conv' in k)
                print(f"  - RRDB blocks: {rrdb_count} 個權重")
                print(f"  - trunk_conv: {trunk_conv_count} 個權重")
            else:
                print("❌ 未找到匹配的 RRDB 權重，使用隨機初始化")
                print("   提示: 請運行 'python diagnose_esrgan.py' 檢查預訓練模型")

        except Exception as e:
            print(f"❌ 加載預訓練權重失敗: {e}")
            print("CCSR-ESRGAN 將使用隨機初始化")
            import traceback
            traceback.print_exc()

    def _freeze_rrdb(self):
        """凍結 RRDB trunk 的參數"""
        for module in self.rrdb_trunk:
            for param in module.parameters():
                param.requires_grad = False

        for param in self.trunk_conv.parameters():
            param.requires_grad = False

        print("RRDB trunk 參數已凍結")

    def forward(self, lr_image: torch.Tensor, view_idx: int) -> torch.Tensor:
        """
        Args:
            lr_image: [B, 3, H, W]，範圍 [-1, 1]
            view_idx: 視角索引

        Returns:
            sr_image: [B, 3, H*scale, W*scale]，範圍 [-1, 1]
        """
        batch_size = lr_image.shape[0]

        # 1. 獲取視角特定的潛在代碼
        latent_code = self.cclc(view_idx)
        latent_code = latent_code.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # 2. 上採樣 LR 圖像到與潛在代碼相同的尺寸
        lr_upsampled = F.interpolate(lr_image, size=latent_code.shape[-2:],
                                     mode='bilinear', align_corners=False)

        # 3. 融合 LR 圖像和潛在代碼
        combined_input = torch.cat([lr_upsampled, latent_code], dim=1)  # [B, 6, H, W]

        # 將輸入從 [-1, 1] 轉換到 [0, 1]（ESRGAN 期望範圍）
        combined_input = (combined_input + 1) / 2.0

        # 4. 融合卷積
        fea = self.fusion_conv(combined_input)
        trunk = fea

        # 5. RRDB trunk
        for rrdb in self.rrdb_trunk:
            trunk = rrdb(trunk)

        trunk = self.trunk_conv(trunk)
        fea = fea + trunk

        # 6. 上採樣 (x4 = 2x + 2x)
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))

        # 7. 高分辨率卷積
        sr_output = self.conv_last(self.lrelu(self.hrconv(fea)))

        # 將輸出從 [0, 1] 轉換回 [-1, 1]
        sr_output = sr_output * 2.0 - 1.0
        sr_output = torch.clamp(sr_output, -1, 1)

        # 8. 應用一致性強制模組 (CEM)
        refined_sr = self.cem(sr_output, lr_image)

        return refined_sr