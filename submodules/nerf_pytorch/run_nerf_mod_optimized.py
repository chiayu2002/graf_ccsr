"""
优化版本的 render_nerfacc
关键改进：只调用一次 estimator.sampling()，而不是每个 batch 调用一次
"""
import torch
from nerfacc import OccGridEstimator, accumulate_along_rays, render_weight_from_density

def render_nerfacc_optimized(H, W, focal, label, rays=None,
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
    优化的 NerfAcc 渲染函数

    关键优化：
    1. 只调用一次 estimator.sampling()（而不是 batch_size 次）
    2. 像原始 render 一样扩展 features
    3. 减少重复的 occupancy grid 查询
    """
    if estimator is None:
        raise ValueError("NerfAcc render requires an OccGridEstimator")

    # 解析 rays
    rays_o, rays_d = rays
    sh = rays_d.shape  # 保存原始形状用于最后 reshape

    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    N_rays_total = rays_o.shape[0]
    device = rays_o.device

    # 处理 viewdirs
    if use_viewdirs:
        viewdirs = rays_d / (torch.norm(rays_d, dim=-1, keepdim=True) + 1e-8)
    else:
        viewdirs = None

    # ========== 扩展 features（像原始 render 一样）==========
    if features is not None:
        bs = features.shape[0]
        N_rays_per_batch = N_rays_total // bs
        # 扩展 features: [bs, feat_dim] -> [bs, N_rays_per_batch, feat_dim] -> [N_rays_total, feat_dim]
        features_expanded = features.unsqueeze(1).expand(-1, N_rays_per_batch, -1).flatten(0, 1)
    else:
        bs = 1
        features_expanded = None

    # ========== 定义查询函数 ==========
    def sigma_fn(t_starts, t_ends, ray_indices):
        """查询密度（用于 estimator.sampling）"""
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0

        N = positions.shape[0]
        pos_input = positions.unsqueeze(0)  # [1, N, 3]

        # 使用平均 viewdir（简化版本）
        if viewdirs is not None:
            vdirs = viewdirs[ray_indices]
            vdir = vdirs.mean(dim=0, keepdim=True)  # [1, 3]
        else:
            vdir = torch.zeros(1, 3, device=device)
            vdir[0, 2] = -1.0

        # 获取对应的 feature
        if features_expanded is not None:
            # 取平均 feature（因为可能一个采样点对应多个 batch）
            feat = features_expanded[ray_indices].mean(dim=0, keepdim=True)  # [1, feat_dim]
        else:
            feat = None

        # 取第一个 label（假设 batch 内 label 相同）
        lbl = label[0:1]

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

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        """查询 RGB 和 sigma（用于渲染）"""
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0

        N = positions.shape[0]
        pos_input = positions.unsqueeze(0)  # [1, N, 3]

        # 使用平均 viewdir
        if viewdirs is not None:
            vdirs = viewdirs[ray_indices]
            vdir = vdirs.mean(dim=0, keepdim=True)  # [1, 3]
        else:
            vdir = torch.zeros(1, 3, device=device)
            vdir[0, 2] = -1.0

        # 获取对应的 feature
        if features_expanded is not None:
            feat = features_expanded[ray_indices].mean(dim=0, keepdim=True)
        else:
            feat = None

        lbl = label[0:1]

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

    # ========== 关键优化：只调用一次 estimator.sampling() ==========
    with torch.no_grad():
        ray_indices, t_starts, t_ends = estimator.sampling(
            rays_o=rays_o,
            rays_d=rays_d,
            sigma_fn=sigma_fn,
            near_plane=near,
            far_plane=far,
            render_step_size=render_step_size,
            early_stop_eps=1e-4,
            alpha_thre=0.001,
            stratified=True,
        )

    # 初始化输出
    rgb_map = torch.zeros(N_rays_total, 3, device=device)
    acc_map = torch.zeros(N_rays_total, device=device)
    depth_map = torch.zeros(N_rays_total, device=device)

    if len(ray_indices) > 0:
        # 查询 RGB 和 sigma
        rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices)

        # 计算权重
        weights, trans, alphas = render_weight_from_density(
            t_starts=t_starts,
            t_ends=t_ends,
            sigmas=sigmas,
            ray_indices=ray_indices,
            n_rays=N_rays_total,
        )

        # 累积颜色
        rgb_map = accumulate_along_rays(
            weights=weights,
            ray_indices=ray_indices,
            values=rgbs,
            n_rays=N_rays_total,
        )

        # 累积不透明度
        acc_map = accumulate_along_rays(
            weights=weights,
            ray_indices=ray_indices,
            values=None,
            n_rays=N_rays_total,
        )
        if acc_map.dim() > 1:
            acc_map = acc_map.reshape(N_rays_total)

        # 累积深度
        depth_map = accumulate_along_rays(
            weights=weights,
            ray_indices=ray_indices,
            values=(t_starts + t_ends)[:, None] / 2.0,
            n_rays=N_rays_total,
        )
        if depth_map.dim() > 1:
            depth_map = depth_map.reshape(N_rays_total)

    # 计算 disparity
    disp_map = 1.0 / torch.clamp(depth_map / (acc_map + 1e-10), min=1e-10)

    # Reshape 回原始形状
    rgb_final = rgb_map.view(list(sh[:-1]) + [3])
    disp_final = disp_map.view(list(sh[:-1]))
    acc_final = acc_map.view(list(sh[:-1]))

    # 统计信息
    extras = {
        'n_samples': len(t_starts),
        'n_rays': N_rays_total,
        'sampling_efficiency': 1 - (len(t_starts) / (N_rays_total * 64)) if N_rays_total > 0 else 0
    }

    return rgb_final, disp_final, acc_final, extras
