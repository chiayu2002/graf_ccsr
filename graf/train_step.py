import torch
# from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd
import torch.nn as nn
import pickle
import numpy as np
import os


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

class RegionWeightedLoss(nn.Module):
    """对图片不同区域应用不同权重的损失

    用于让模型更关注图片下半部分的细节
    """

    def __init__(self, lower_half_weight=2.0, upper_half_weight=1.0):
        """
        Args:
            lower_half_weight: 下半部分的损失权重
            upper_half_weight: 上半部分的损失权重
        """
        super().__init__()
        self.lower_half_weight = lower_half_weight
        self.upper_half_weight = upper_half_weight

    def forward(self, d_out, target, height_indices, H):
        """
        计算区域加权的损失

        Args:
            d_out: 判别器输出 [B*N]
            target: 目标值 (0 或 1)
            height_indices: 每个样本的高度索引 [B*N]
            H: 图片高度

        Returns:
            加权后的损失
        """
        mid_point = H // 2

        # 创建权重 mask
        weights = torch.ones_like(d_out, dtype=torch.float32)

        # 对每个判别器输出元素，根据其对应的高度索引分配权重
        for i in range(d_out.shape[0]):
            if i < len(height_indices):
                if height_indices[i] >= mid_point:
                    weights[i] = self.lower_half_weight
                else:
                    weights[i] = self.upper_half_weight

        # 计算 BCE 损失
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(d_out, targets, reduction='none')

        # 应用权重
        weighted_loss = (bce_loss * weights).mean()

        return weighted_loss


class CCSRNeRFLoss(nn.Module):
    """CCSR與NeRF的聯合損失"""

    def __init__(self, alpha_init=1.0, alpha_decay=0.0001):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.alpha_init = alpha_init
        self.alpha_decay = alpha_decay
        self.iteration = 0

    def forward(self, ccsr_output, nerf_output):
        """
        計算CCSR輸出與NeRF輸出的MSE損失

        Args:
            ccsr_output: CCSR生成的圖像 [B, N_samples, 3]
            nerf_output: NeRF渲染的圖像 [B, N_samples, 3]
        """
        # 動態調整權重
        alpha = self.alpha_init * np.exp(-self.alpha_decay * self.iteration)

        # 計算MSE損失
        consistency_loss = self.mse_loss(ccsr_output, nerf_output.detach())

        self.iteration += 1
        return alpha * consistency_loss
    

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