"""
ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks)
Architecture for RRDB_ESRGAN_x4 model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 明确导出的类
__all__ = ['ResidualDenseBlock', 'RRDB', 'RRDBNet', 'ESRGANWrapper']


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block (RDB) used in RRDB"""

    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock, self).__init__()
        # 5個卷積層
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock(nf, gc)
        self.RDB2 = ResidualDenseBlock(nf, gc)
        self.RDB3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RRDB Network for ESRGAN"""

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        """
        Args:
            in_nc: 輸入通道數 (3 for RGB)
            out_nc: 輸出通道數 (3 for RGB)
            nf: 特徵圖通道數
            nb: RRDB blocks 數量
            gc: growth channel (Dense block 中的通道數)
            scale: 上採樣倍率
        """
        super(RRDBNet, self).__init__()
        self.scale = scale

        # 第一層卷積
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)

        # RRDB blocks
        self.RRDB_trunk = nn.ModuleList()
        for _ in range(nb):
            self.RRDB_trunk.append(RRDB(nf, gc))

        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # 上採樣
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # 最後的卷積層
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # 第一層特徵提取
        fea = self.conv_first(x)
        trunk = fea

        # RRDB blocks
        for rrdb in self.RRDB_trunk:
            trunk = rrdb(trunk)

        trunk = self.trunk_conv(trunk)
        fea = fea + trunk

        # 上採樣 (x4 = 2x upsample twice)
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))

        # 最後的卷積
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out


class ESRGANWrapper(nn.Module):
    """
    ESRGAN 包裝類，用於整合到 GRAF-CCSR 中
    處理輸入格式轉換和預訓練模型加載
    """

    def __init__(self, pretrained_path=None, freeze=True, scale=4):
        """
        Args:
            pretrained_path: 預訓練模型權重路徑
            freeze: 是否凍結 ESRGAN 參數
            scale: 上採樣倍率
        """
        super(ESRGANWrapper, self).__init__()

        # 創建 RRDB 網絡
        self.model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=scale)
        self.scale = scale

        # 加載預訓練權重
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

        # 凍結參數
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def load_pretrained(self, pretrained_path):
        """加載預訓練權重"""
        import os
        if not os.path.exists(pretrained_path):
            print(f"警告: 預訓練模型文件不存在: {pretrained_path}")
            print("將使用隨機初始化的權重")
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

            self.model.load_state_dict(pretrained_dict, strict=True)
            print(f"成功加載預訓練模型: {pretrained_path}")
        except Exception as e:
            print(f"加載預訓練模型失敗: {e}")
            print("將使用隨機初始化的權重")

    def forward(self, x):
        """
        Args:
            x: 輸入張量 [B, C, H, W]，範圍 [-1, 1] (GRAF 格式)
        Returns:
            超分辨率圖像 [B, C, H*scale, W*scale]，範圍 [-1, 1]
        """
        # 將輸入從 [-1, 1] 轉換到 [0, 1] (ESRGAN 期望的格式)
        x_esrgan = (x + 1) / 2.0

        # ESRGAN 推理
        with torch.set_grad_enabled(self.training):
            sr = self.model(x_esrgan)

        # 將輸出從 [0, 1] 轉換回 [-1, 1] (GRAF 格式)
        sr = sr * 2.0 - 1.0

        # 確保輸出在合理範圍內
        sr = torch.clamp(sr, -1, 1)

        return sr

    def train(self, mode=True):
        """重寫 train 方法，確保凍結時保持 eval 模式"""
        if hasattr(self, 'model'):
            # 檢查是否所有參數都被凍結
            frozen = all(not p.requires_grad for p in self.model.parameters())
            if frozen:
                self.model.eval()
            else:
                self.model.train(mode)
        return super().train(mode)
