import torch
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd
import torch.nn as nn
import pickle
import numpy as np
import os





def compute_loss(d_outs, target):

        d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
        loss = 0

        for d_out in d_outs:
            targets = d_out.new_full(size=d_out.size(), fill_value=target)
            loss += F.binary_cross_entropy_with_logits(d_out, targets)
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
