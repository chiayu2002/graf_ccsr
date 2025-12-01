import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from functools import partial

import matplotlib.pyplot as plt

from .run_nerf_helpers_mod import *

# ========== NerfAcc imports ==========
import nerfacc
from nerfacc import OccGridEstimator, render_weight_from_density, accumulate_along_rays

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

relu = partial(F.relu, inplace=True)


def batchify(fn, chunk):
    if chunk is None:
        return fn
    def ret(inputs, label):
        label_oftype = label[:,0]
        return torch.cat([fn(inputs[i:i+chunk], label_oftype) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, label, embed_fn, embeddirs_fn, features=None, netchunk=1024*64):
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)
    
    if features is not None:
        features = features.unsqueeze(1).expand(-1, inputs.shape[1], -1).flatten(0, 1)
        features_shape = features
        embedded = torch.cat([embedded, features_shape], -1)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded, label)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, label, chunk=1024*32, **kwargs):
    all_ret = {}
    features = kwargs.get('features')
    for i in range(0, rays_flat.shape[0], chunk):
        if features is not None:
            kwargs['features'] = features[i:i+chunk]
        ret = render_rays(rays_flat[i:i+chunk], label, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, label, chunk=1024*32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """原本的渲染函數（不使用 NerfAcc）"""
    
    if c2w is not None:
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        rays_o, rays_d = rays

    if use_viewdirs:
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    if kwargs.get('features') is not None:
        bs = kwargs['features'].shape[0]
        N_rays = sh[0] // bs
        kwargs['features'] = kwargs['features'].unsqueeze(1).expand(-1, N_rays, -1).flatten(0, 1)

    all_ret = batchify_rays(rays, label, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


# ========== NerfAcc 渲染函數 ==========

def render_nerfacc(H, W, focal, label, rays=None,
                   near=0., far=1.,
                   use_viewdirs=True,
                   estimator=None,
                   render_step_size=5e-3,
                   network_fn=None,
                   network_query_fn=None,
                   features=None,
                   network_fine=None,
                   **kwargs):
    """
    使用 NerfAcc 加速的渲染函數
    逐批次處理，保持與 GRAF 網路的相容性
    """
    if estimator is None:
        raise ValueError("NerfAcc render requires an OccGridEstimator")
    
    # 解析 rays
    rays_o, rays_d = rays
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()
    
    N_rays_total = rays_o.shape[0]
    device = rays_o.device
    
    # 計算 batch 資訊
    if features is not None:
        bs = features.shape[0]
        n_rays_per_batch = N_rays_total // bs
    else:
        bs = 1
        n_rays_per_batch = N_rays_total
    
    # 準備輸出
    all_rgb = []
    all_disp = []
    all_acc = []
    total_samples = 0
    
    # ========== 逐批次處理 ==========
    for b in range(bs):
        # 提取這個 batch 的 rays
        start_idx = b * n_rays_per_batch
        end_idx = (b + 1) * n_rays_per_batch
        
        batch_rays_o = rays_o[start_idx:end_idx]  # [n_rays, 3]
        batch_rays_d = rays_d[start_idx:end_idx]  # [n_rays, 3]
        
        # 這個 batch 的 feature
        if features is not None:
            batch_feature = features[b:b+1]  # [1, feat_dim]
        else:
            batch_feature = None
        
        # 這個 batch 的 label
        batch_label = label[b:b+1]  # [1, ...]
        
        # 處理 viewdirs
        if use_viewdirs:
            batch_viewdirs = batch_rays_d / (torch.norm(batch_rays_d, dim=-1, keepdim=True) + 1e-8)
        else:
            batch_viewdirs = None
        
        N_rays = batch_rays_o.shape[0]
        
        # ========== 定義這個 batch 的查詢函數 ==========
        def make_query_fn(feat, lbl, batch_vdirs):
            def query_sigma(positions, ray_idx):
                """查詢密度 - 使用對應ray的真實viewdir"""
                N = positions.shape[0]
                pos_input = positions.unsqueeze(0)  # [1, N, 3]

                # 使用對應的viewdirs（每個position對應其ray的viewdir）
                if batch_vdirs is not None:
                    vdirs = batch_vdirs[ray_idx]  # [N, 3]
                    # 取平均viewdir作為代表（因為network_query_fn需要[1, 3]）
                    vdir = vdirs.mean(dim=0, keepdim=True)  # [1, 3]
                else:
                    vdir = torch.zeros(1, 3, device=device)
                    vdir[0, 2] = -1.0

                with torch.no_grad():
                    raw = network_query_fn(
                        pos_input,
                        vdir,
                        network_fn,
                        lbl,
                        feat
                    )  # [1, N, 4]

                sigmas = torch.relu(raw[0, :, 3])  # [N]
                return sigmas

            def query_rgb_sigma(positions, viewdirs_sample):
                """查詢 RGB 和 sigma - 使用真實的viewdirs"""
                N = positions.shape[0]
                pos_input = positions.unsqueeze(0)  # [1, N, 3]

                # 使用傳入的真實viewdirs
                if viewdirs_sample is not None and len(viewdirs_sample) > 0:
                    # 取平均viewdir（因為GRAF的network_query_fn接受[1, 3]的viewdir）
                    vdir = viewdirs_sample.mean(dim=0, keepdim=True)  # [1, 3]
                else:
                    vdir = torch.zeros(1, 3, device=device)
                    vdir[0, 2] = -1.0

                raw = network_query_fn(
                    pos_input,
                    vdir,
                    network_fn,
                    lbl,
                    feat
                )  # [1, N, 4]

                rgbs = torch.sigmoid(raw[0, :, :3])  # [N, 3]
                sigmas = torch.relu(raw[0, :, 3])  # [N]
                return rgbs, sigmas

            return query_sigma, query_rgb_sigma
        
        query_sigma, query_rgb_sigma = make_query_fn(batch_feature, batch_label, batch_viewdirs)

        # ========== sigma_fn for NerfAcc ==========
        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = batch_rays_o[ray_indices]
            t_dirs = batch_rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            return query_sigma(positions, ray_indices)
        
        # ========== NerfAcc 射線採樣 ==========
        with torch.no_grad():
            ray_indices, t_starts, t_ends = estimator.sampling(
                rays_o=batch_rays_o,
                rays_d=batch_rays_d,
                sigma_fn=sigma_fn,
                near_plane=near,
                far_plane=far,
                render_step_size=render_step_size,
                early_stop_eps=1e-4,
                alpha_thre=0.0,
                stratified=True,
            )
        
        # ========== 渲染這個 batch ==========
        if len(ray_indices) > 0:
            # 計算採樣點位置
            t_origins = batch_rays_o[ray_indices]
            t_dirs = batch_rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            
            if batch_viewdirs is not None:
                vdirs = batch_viewdirs[ray_indices]
            else:
                vdirs = None
            
            # 查詢 RGB 和 sigma
            rgbs, sigmas = query_rgb_sigma(positions, vdirs)
            
            # 計算權重
            weights, trans, alphas = render_weight_from_density(
                t_starts=t_starts,
                t_ends=t_ends,
                sigmas=sigmas,
                ray_indices=ray_indices,
                n_rays=N_rays,
            )
            
            # 累積顏色
            rgb_map = accumulate_along_rays(
                weights=weights,
                ray_indices=ray_indices,
                values=rgbs,
                n_rays=N_rays,
            )
            
            # 累積不透明度
            acc_map = accumulate_along_rays(
                weights=weights,
                ray_indices=ray_indices,
                values=None,
                n_rays=N_rays,
            )
            
            # 累積深度
            depth_map = accumulate_along_rays(
                weights=weights,
                ray_indices=ray_indices,
                values=(t_starts + t_ends)[:, None] / 2.0,
                n_rays=N_rays,
            ).squeeze(-1)
            
            total_samples += len(t_starts)
        else:
            rgb_map = torch.zeros(N_rays, 3, device=device)
            acc_map = torch.zeros(N_rays, device=device)
            depth_map = torch.zeros(N_rays, device=device)
        
        # 計算 disparity
        disp_map = 1.0 / torch.clamp(depth_map / (acc_map + 1e-10), min=1e-10)
        
        all_rgb.append(rgb_map)
        all_disp.append(disp_map)
        all_acc.append(acc_map)
    
    # 合併所有 batch 的結果
    rgb_final = torch.cat(all_rgb, dim=0)  # [N_rays_total, 3]
    disp_final = torch.cat(all_disp, dim=0)  # [N_rays_total]
    acc_final = torch.cat(all_acc, dim=0)  # [N_rays_total]
    
    extras = {
        'n_samples': total_samples,
    }
    
    return [rgb_final, disp_final, acc_final, extras]


def create_nerf(args):
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch += args.feat_dim
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, numclasses=args.num_class)
    grad_vars = list(model.parameters())
    named_params = list(model.named_parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, numclasses=args.num_class)
        grad_vars += list(model_fine.parameters())
        named_params += list(model_fine.named_parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn, label, features: run_network(inputs, viewdirs, network_fn, label,
                                                                                  features=features,
                                                                                  embed_fn=embed_fn,
                                                                                  embeddirs_fn=embeddirs_fn,
                                                                                  netchunk=args.netchunk)

    render_kwargs_train = {             
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'raw_noise_std' : args.raw_noise_std,
        'ndc': False,
        'lindisp': False,
    }

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, grad_vars, named_params


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, pytest=False):
    raw2alpha = lambda raw, dists, act_fn=relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / (torch.sum(weights, -1)+1e-10))
    acc_map = torch.sum(weights, -1)

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                label,
                network_fn,
                network_query_fn,
                N_samples,
                features=None,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6]
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        t_rand = torch.rand(z_vals.shape)
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]

    raw = network_query_fn(pts, viewdirs, network_fn, label, features)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, pytest=pytest)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn, label, features)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}

    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def save_rays_torch(rays_o, rays_d, save_path="rays_data.pt"):
    rays_data = {
        "rays_o": rays_o,
        "rays_d": rays_d
    }
    torch.save(rays_data, save_path)
    print(f"射線數據已儲存至 {save_path}")