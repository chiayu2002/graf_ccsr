# ========== 修复版 render_nerfacc ==========
# 替换 submodules/nerf_pytorch/run_nerf_mod.py 中的 render_nerfacc 函数（line 120-353）

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
    ⚡ 優化版本：只調用一次 estimator.sampling()（而不是 batch_size 次）
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

    # 計算 batch 資訊
    if features is not None:
        bs = features.shape[0]
        n_rays_per_batch = N_rays_total // bs
    else:
        bs = 1
        n_rays_per_batch = N_rays_total

    # 處理 viewdirs（所有 rays）
    if use_viewdirs:
        viewdirs = rays_d / (torch.norm(rays_d, dim=-1, keepdim=True) + 1e-8)
    else:
        viewdirs = None

    # ========== ⚡ 關鍵優化：只調用一次 estimator.sampling() ==========
    # 定義統一的 sigma_fn（使用第一個 batch 的 feature 作為代表）
    def sigma_fn_unified(t_starts, t_ends, ray_indices):
        """統一的密度查詢函數（用於所有 rays 的採樣）"""
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

        # 使用第一個 batch 的 feature（occupancy 主要依賴幾何，對 feature 不敏感）
        feat = features[0:1] if features is not None else None
        lbl = label[0:1]

        with torch.no_grad():
            raw = network_query_fn(pos_input, vdir, network_fn, lbl, feat)

        sigmas = torch.relu(raw[0, :, 3])  # [N]
        return sigmas

    # ⭐ 只調用一次 estimator.sampling()（之前是調用 bs=8 次！）
    with torch.no_grad():
        ray_indices_all, t_starts_all, t_ends_all = estimator.sampling(
            rays_o=rays_o,  # 所有 rays
            rays_d=rays_d,
            sigma_fn=sigma_fn_unified,
            near_plane=near,
            far_plane=far,
            render_step_size=render_step_size,
            early_stop_eps=1e-4,
            alpha_thre=0.001,
            stratified=True,
        )

    # 準備輸出
    all_rgb = []
    all_disp = []
    all_acc = []
    total_samples = 0

    # ========== 按 batch 處理渲染（但不重新採樣）==========
    for b in range(bs):
        # 提取這個 batch 的 rays 範圍
        batch_start = b * n_rays_per_batch
        batch_end = (b + 1) * n_rays_per_batch

        # 篩選屬於這個 batch 的採樣點
        mask = (ray_indices_all >= batch_start) & (ray_indices_all < batch_end)
        ray_indices = ray_indices_all[mask] - batch_start  # 轉換為 batch 內索引
        t_starts = t_starts_all[mask]
        t_ends = t_ends_all[mask]

        # 這個 batch 的 rays
        batch_rays_o = rays_o[batch_start:batch_end]
        batch_rays_d = rays_d[batch_start:batch_end]
        N_rays = n_rays_per_batch

        # 這個 batch 的 feature 和 label
        if features is not None:
            batch_feature = features[b:b+1]
        else:
            batch_feature = None
        batch_label = label[b:b+1]

        # 這個 batch 的 viewdirs
        if viewdirs is not None:
            batch_viewdirs = viewdirs[batch_start:batch_end]
        else:
            batch_viewdirs = None

        # 初始化輸出
        rgb_map = torch.zeros(N_rays, 3, device=device)
        acc_map = torch.zeros(N_rays, device=device)
        depth_map = torch.zeros(N_rays, device=device)

        if len(ray_indices) > 0:
            # 計算採樣點位置
            t_origins = batch_rays_o[ray_indices]
            t_dirs = batch_rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0

            # 準備 viewdirs
            if batch_viewdirs is not None:
                vdirs = batch_viewdirs[ray_indices]
                vdir = vdirs.mean(dim=0, keepdim=True)  # [1, 3]
            else:
                vdir = torch.zeros(1, 3, device=device)
                vdir[0, 2] = -1.0

            # 查詢 RGB 和 sigma（使用這個 batch 的真實 feature）
            N = positions.shape[0]
            pos_input = positions.unsqueeze(0)  # [1, N, 3]

            raw = network_query_fn(
                pos_input,
                vdir,
                network_fn,
                batch_label,
                batch_feature
            )  # [1, N, 4]

            rgbs = torch.sigmoid(raw[0, :, :3])  # [N, 3]
            sigmas = torch.relu(raw[0, :, 3])  # [N]

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
            if acc_map.dim() > 1:
                acc_map = acc_map.reshape(N_rays)

            # 累積深度
            depth_map = accumulate_along_rays(
                weights=weights,
                ray_indices=ray_indices,
                values=(t_starts + t_ends)[:, None] / 2.0,
                n_rays=N_rays,
            )
            if depth_map.dim() > 1:
                depth_map = depth_map.reshape(N_rays)

            total_samples += len(t_starts)

        # 計算 disparity
        disp_map = 1.0 / torch.clamp(depth_map / (acc_map + 1e-10), min=1e-10)

        all_rgb.append(rgb_map)
        all_disp.append(disp_map)
        all_acc.append(acc_map)

    # 合併所有 batch 的結果
    rgb_final = torch.cat(all_rgb, dim=0)  # [N_rays_total, 3]
    disp_final = torch.cat(all_disp, dim=0)  # [N_rays_total]
    acc_final = torch.cat(all_acc, dim=0)  # [N_rays_total]

    # Reshape 回原始形状
    rgb_final = rgb_final.view(list(sh[:-1]) + [3])
    disp_final = disp_final.view(list(sh[:-1]))
    acc_final = acc_final.view(list(sh[:-1]))

    # 計算統計信息
    theoretical_samples = N_rays_total * int((far - near) / render_step_size)
    sample_reduction_pct = (1 - total_samples / theoretical_samples) * 100 if theoretical_samples > 0 else 0

    extras = {
        'n_samples': total_samples,
        'n_rays': N_rays_total,
        'theoretical_samples': theoretical_samples,
        'sample_reduction': sample_reduction_pct,
    }

    return [rgb_final, disp_final, acc_final, extras]
