import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
import warnings
from typing import Optional, Union


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
    

class ResidualDenseBlock(nn.Module):
    """ESRGAN 的 Residual Dense Block (RDB)"""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block (RRDB)"""

    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """ESRGAN 的 RRDB 網絡架構"""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                 num_grow_ch=32, scale=4):
        super().__init__()
        self.scale = scale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # Upsample x2
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        # Upsample x2
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class ESRGANWrapper(nn.Module):
    """ESRGAN 模型的包裝器，支持條件輸入和潛在代碼"""

    def __init__(self, pretrained_path: Optional[str] = None,
                 num_in_ch: int = 6,  # 3(image) + 3(latent code)
                 scale: int = 4,
                 num_block: int = 23):
        super().__init__()

        self.scale = scale
        self.esrgan = RRDBNet(num_in_ch=num_in_ch, num_out_ch=3,
                             num_feat=64, num_block=num_block,
                             num_grow_ch=32, scale=scale)

        # 如果提供預訓練權重路徑，嘗試加載
        if pretrained_path and os.path.exists(pretrained_path):
            self.load_pretrained(pretrained_path)
        elif pretrained_path:
            warnings.warn(f"預訓練模型路徑 {pretrained_path} 不存在，使用隨機初始化")

    def load_pretrained(self, model_path: str):
        """加載預訓練的 ESRGAN 權重"""
        try:
            state_dict = torch.load(model_path, map_location='cpu')

            # 處理不同的權重格式
            if 'params_ema' in state_dict:
                state_dict = state_dict['params_ema']
            elif 'params' in state_dict:
                state_dict = state_dict['params']
            elif 'model' in state_dict:
                state_dict = state_dict['model']

            # 調整輸入通道數（如果預訓練模型是 3 通道輸入）
            if 'conv_first.weight' in state_dict:
                pretrained_first_conv = state_dict['conv_first.weight']
                if pretrained_first_conv.shape[1] != self.esrgan.conv_first.weight.shape[1]:
                    # 擴展第一層卷積以接受更多通道
                    new_weight = torch.zeros_like(self.esrgan.conv_first.weight)
                    new_weight[:, :3, :, :] = pretrained_first_conv  # 複製前3個通道
                    new_weight[:, 3:, :, :] = pretrained_first_conv.mean(dim=1, keepdim=True)  # 其餘通道用平均值初始化
                    state_dict['conv_first.weight'] = new_weight
                    warnings.warn("已調整第一層卷積以接受額外的潛在代碼通道")

            # 加載權重（允許部分匹配）
            self.esrgan.load_state_dict(state_dict, strict=False)
            print(f"成功加載預訓練 ESRGAN 模型: {model_path}")

        except Exception as e:
            warnings.warn(f"加載預訓練模型失敗: {e}，使用隨機初始化")

    def forward(self, x):
        return self.esrgan(x)


class CCSR(nn.Module):
    """一致性控制超分辨率模組 (使用 ESRGAN)"""

    def __init__(self, num_views: int, lr_height: int, lr_width: int,
                 scale_factor: int = 4,
                 use_esrgan: bool = True,
                 esrgan_path: Optional[str] = None,
                 num_rrdb_blocks: int = 23):
        """
        Args:
            num_views: 視角數量
            lr_height: 低分辨率圖像高度
            lr_width: 低分辨率圖像寬度
            scale_factor: 超分辨率放大倍數
            use_esrgan: 是否使用 ESRGAN（否則使用簡單網絡）
            esrgan_path: 預訓練 ESRGAN 模型路徑
            num_rrdb_blocks: RRDB 塊數量（越多越好但越慢，推薦: 6-23）
        """
        super().__init__()

        self.scale_factor = scale_factor
        self.use_esrgan = use_esrgan

        # 一致性控制潛在代碼
        self.cclc = ConsistencyControllingLatentCode(num_views, lr_height, lr_width, scale_factor)

        # 一致性執行模組
        self.cem = ConsistencyEnforcingModule()

        # 超分辨率網絡
        if use_esrgan:
            self.sr_network = ESRGANWrapper(
                pretrained_path=esrgan_path,
                num_in_ch=6,  # 3(image) + 3(latent)
                scale=scale_factor,
                num_block=num_rrdb_blocks
            )
        else:
            # 簡化的超分辨率網絡（原始版本）
            self.sr_network = self._build_simple_sr_network(scale_factor)

        self.activation = nn.LeakyReLU(0.2)

    def _build_simple_sr_network(self, scale_factor: int) -> nn.Module:
        """構建簡化的超分辨率網絡（備用方案）"""
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
        """
        CCSR前向傳播

        Args:
            lr_image: 低分辨率圖像 [B, 3, H, W]，範圍 [-1, 1]
            view_idx: 視角索引

        Returns:
            refined_sr: 超分辨率圖像 [B, 3, H*scale, W*scale]，範圍 [-1, 1]
        """
        batch_size = lr_image.shape[0]
        device = lr_image.device

        # 獲取潛在代碼並擴展到batch
        latent_code = self.cclc(view_idx)
        latent_code = latent_code.to(device)  # 確保設備一致
        latent_code = latent_code.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # 上採樣低分辨率圖像以匹配潛在代碼的尺寸
        lr_upsampled = F.interpolate(lr_image, size=latent_code.shape[-2:],
                                    mode='bilinear', align_corners=False)

        # 連接輸入：lr_image + latent_code
        combined_input = torch.cat([lr_upsampled, latent_code], dim=1)

        # 生成超分辨率圖像
        sr_output = self.sr_network(combined_input)

        # 如果不使用ESRGAN，需要額外的激活和裁剪
        if not self.use_esrgan:
            sr_output = self.activation(sr_output)

        sr_output = torch.clamp(sr_output, -1, 1)  # 匹配GRAF的圖像範圍 [-1, 1]

        # 應用一致性執行模組
        refined_sr = self.cem(sr_output, lr_image)
        refined_sr = torch.clamp(refined_sr, -1, 1)

        return refined_sr