import torch
# from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import os
import torchvision.models as models


class MCE_Loss(nn.Module):
    def __init__(self):
        super(MCE_Loss, self).__init__()

    def __call__(self, n_each_task, output, target):
        loss = []
        for i, n in enumerate(n_each_task):
            if i == 0:
                loss.append(nn.CrossEntropyLoss()(output[:, :n], target[:, :n]))
            else:
                summation = sum(n_each_task[:i])
                loss.append(nn.CrossEntropyLoss()(output[:, summation:summation+n], target[:, summation:summation+n]))
        
        return sum(loss)

class VGGPerceptualLoss(nn.Module):
    """VGG感知損失用於保持高頻細節"""

    def __init__(self, device='cuda'):
        super().__init__()
        # 使用預訓練的VGG19提取特徵
        vgg = models.vgg19(pretrained=True).features
        self.feature_extractor = nn.Sequential(*list(vgg[:36])).eval().to(device)

        # 凍結VGG參數
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # 用於標準化輸入
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        """將[-1,1]範圍的圖像轉換為VGG期望的[0,1]範圍並標準化"""
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        """計算感知損失"""
        pred_normalized = self.normalize(pred)
        target_normalized = self.normalize(target)

        pred_features = self.feature_extractor(pred_normalized)
        target_features = self.feature_extractor(target_normalized)

        loss = F.mse_loss(pred_features, target_features)
        return loss


class CCSRNeRFLoss(nn.Module):
    """CCSR與NeRF的聯合損失 - 增強版本包含感知損失"""

    def __init__(self, alpha_init=1.0, alpha_decay=0.0001, perceptual_weight=0.1, device='cuda'):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = VGGPerceptualLoss(device=device)
        self.alpha_init = alpha_init
        self.alpha_decay = alpha_decay
        self.perceptual_weight = perceptual_weight
        self.iteration = 0

    def forward(self, ccsr_output, nerf_output):
        """
        計算CCSR輸出與NeRF輸出的聯合損失

        Args:
            ccsr_output: CCSR生成的圖像 [B, 3, H, W] 或 [B*H*W, 3]
            nerf_output: NeRF渲染的圖像 [B, 3, H, W] 或 [B*H*W, 3]
        """
        # 動態調整權重
        alpha = self.alpha_init * np.exp(-self.alpha_decay * self.iteration)

        # 計算MSE損失（像素級）
        mse_loss = self.mse_loss(ccsr_output, nerf_output.detach())

        # 計算感知損失（特徵級）
        # 確保輸入是4D張量 [B, C, H, W]
        if ccsr_output.dim() == 2:  # [N, 3]
            # 需要重塑為圖像格式
            batch_size = int(np.sqrt(ccsr_output.shape[0]))
            if batch_size * batch_size == ccsr_output.shape[0]:
                ccsr_img = ccsr_output.view(batch_size, batch_size, 3).permute(2, 0, 1).unsqueeze(0)
                nerf_img = nerf_output.view(batch_size, batch_size, 3).permute(2, 0, 1).unsqueeze(0)
                perceptual_loss = self.perceptual_loss(ccsr_img, nerf_img.detach())
            else:
                perceptual_loss = 0.0
        else:
            perceptual_loss = self.perceptual_loss(ccsr_output, nerf_output.detach())

        # 總損失
        total_loss = alpha * (mse_loss + self.perceptual_weight * perceptual_loss)

        self.iteration += 1
        return total_loss
    

def compute_loss(d_outs, target):

    d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
    loss = 0
    BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

    for d_out in d_outs:
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        # loss += F.binary_cross_entropy_with_logits(d_out, targets)
        loss += BCEWithLogitsLoss(d_out, targets)
        # loss += (2*target - 1) * d_out.mean()
        # floss = loss / len(d_outs)
    return loss / len(d_outs)


def compute_grad2(d_outs, x_in):
    d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
    reg = 0
    for d_out in d_outs:
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg += grad_dout2.view(batch_size, -1).sum(1)
    return reg / len(d_outs)

def wgan_gp_reg(discriminator, x_real, x_fake, y, center=1.):
        batch_size = y.size(0)
        device = torch.device("cuda:0")
        y = y.to(device)

        samples_per_batch = x_real.size(0) // batch_size
        x_real_batched = x_real.reshape(batch_size, samples_per_batch, 3)
        x_fake_batched = x_fake.reshape(batch_size, samples_per_batch, 3)

        eps = torch.rand(batch_size, device=y.device).view(batch_size, 1, 1)
        x_interp_batched = (1 - eps) * x_real_batched + eps * x_fake_batched
        x_interp = x_interp_batched.reshape(-1, 3)
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = discriminator(x_interp, y)

        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()

        return reg

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def save_data(label, rays, iteration, save_dir='./saved_data'):
    """
    簡單的函數用於儲存標籤和光線
    """
    save_dir = os.path.join(save_dir, f'iter_{iteration}')
    os.makedirs(save_dir, exist_ok=True)
    
    # 儲存為 numpy 格式
    label_np = label.detach().cpu().numpy() if isinstance(label, torch.Tensor) else label
    rays_np = rays.detach().cpu().numpy() if isinstance(rays, torch.Tensor) else rays
    
    np.save(os.path.join(save_dir, 'labels.npy'), label_np)
    np.save(os.path.join(save_dir, 'rays.npy'), rays_np)

    with open(os.path.join(save_dir, 'rays_values.csv'), 'w') as f:
        f.write("batch,index,x,y,z\n")  # CSV 標頭
        for batch_idx in range(rays_np.shape[0]):
            for ray_idx in range(rays_np.shape[1]):
                x, y, z = rays_np[batch_idx, ray_idx]
                f.write(f"{batch_idx},{ray_idx},{x},{y},{z}\n")

    with open(os.path.join(save_dir, 'labels_full.txt'), 'w') as f:
        # 設置 numpy 顯示選項以顯示所有元素
        np.set_printoptions(threshold=np.inf, precision=8, suppress=True)
        f.write("Labels (Shape: {}):\n".format(label_np.shape))
        f.write(np.array2string(label_np))

    # 恢復 numpy 的默認顯示選項
    np.set_printoptions(threshold=1000)