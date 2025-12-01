import numpy as np
import torch
import torch.nn.functional as F
from functools import partial

# NerfAcc imports
import nerfacc
from nerfacc import OccGridEstimator

from ..utils import sample_on_sphere, look_at, to_sphere 
from graf.transforms import ImgToPatch
from ..transforms import FullRaySampler
from submodules.nerf_pytorch.run_nerf_mod import render, render_nerfacc
from graf.models.ccsr import CCSR
import os
import pickle


class Generator(object):
    def __init__(self, H, W, focal, radius, ray_sampler, render_kwargs_train, render_kwargs_test, parameters, named_parameters,
                 range_u=(0,1), range_v=(0.01,0.49), v=0, chunk=None, device='cuda', orthographic=False, use_default_rays=False, 
                 use_ccsr=True, num_views=8,
                 # ========== NerfAcc 參數 ==========
                 use_nerfacc=True,
                 near=1.5,
                 far=4.5,
                 nerfacc_resolution=128,
                 render_step_size=None):
        
        self.device = device
        self.H = int(H)
        self.W = int(W)
        self.focal = focal
        self.radius = radius
        self.range_u = range_u
        self.range_v = range_v
        self.chunk = chunk
        self.v = v
        self.use_default_rays = use_default_rays
        self.use_ccsr = use_ccsr
        
        # ========== NerfAcc 設定 ==========
        self.use_nerfacc = use_nerfacc
        self.near = near
        self.far = far
        
        coords = torch.from_numpy(np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), -1))
        self.coords = coords.view(-1, 2)

        self.ray_sampler = ray_sampler
        self.val_ray_sampler = FullRaySampler(orthographic=orthographic)
        self.render_kwargs_train = render_kwargs_train
        self.render_kwargs_test = render_kwargs_test
        self.initial_raw_noise_std = self.render_kwargs_train['raw_noise_std']
        self._parameters = parameters
        self._named_parameters = named_parameters
        self.module_dict = {'generator': self.render_kwargs_train['network_fn']}
        for name, module in [('generator_fine', self.render_kwargs_train['network_fine'])]:
            if module is not None:
                self.module_dict[name] = module

        # ========== 初始化 NerfAcc OccGridEstimator ==========
        if self.use_nerfacc:
            # 根據場景設定 AABB（場景邊界框）
            aabb_scale = radius * 1.5
            scene_aabb = torch.tensor([
                -aabb_scale, -aabb_scale, -aabb_scale,
                 aabb_scale,  aabb_scale,  aabb_scale
            ], dtype=torch.float32, device=device)
            
            self.estimator = OccGridEstimator(
                roi_aabb=scene_aabb,
                resolution=nerfacc_resolution,
                levels=1
            ).to(device)
            
            # 渲染步長
            if render_step_size is None:
                N_samples = render_kwargs_train.get('N_samples', 64)
                self.render_step_size = (far - near) / N_samples
            else:
                self.render_step_size = render_step_size
                
            print(f"[NerfAcc] Initialized:")
            print(f"  - AABB: [{-aabb_scale:.2f}, {aabb_scale:.2f}]^3")
            print(f"  - Resolution: {nerfacc_resolution}")
            print(f"  - Render step size: {self.render_step_size:.4f}")
            print(f"  - Near: {near}, Far: {far}")
        else:
            self.estimator = None
            self.render_step_size = None

        # 添加 CCSR 模組
        if self.use_ccsr:
            lr_height, lr_width = H // 4, W // 4
            self.ccsr = CCSR(num_views=num_views, lr_height=lr_height, lr_width=lr_width, scale_factor=4).to(device)
            self.module_dict['ccsr'] = self.ccsr
            
        for name, module in self.module_dict.items():
            if name in ['generator', 'generator_fine']:
                continue
            self._parameters += list(module.parameters())
            self._named_parameters += list(module.named_parameters())    

        self.parameters = lambda: self._parameters
        self.named_parameters = lambda: self._named_parameters

        self.use_test_kwargs = False
        
        # 設定原始 render 函數
        self.render = partial(render, H=self.H, W=self.W, focal=self.focal, chunk=self.chunk)

    def __call__(self, z, label, rays=None, return_ccsr_output=False):
        bs = z.shape[0]
        if rays is None:
            if self.use_default_rays:
                rays = torch.cat([self.sample_rays() for _ in range(bs)], dim=1)
            else:
                all_rays = []
                v_list = [float(x.strip()) for x in self.v.split(",")]

                for i in range(label.size(0)):
                    second_value = label[i, 1].item()
                    index = int(label[i, 2].item())

                    selected_u = index / 360
                    selected_v = v_list[int(second_value)]

                    rays = self.sample_select_rays(selected_u, selected_v)
                    all_rays.append(rays)
                    
                rays = torch.cat(all_rays, dim=1)

        render_kwargs = self.render_kwargs_test if self.use_test_kwargs else self.render_kwargs_train
        render_kwargs = dict(render_kwargs)  # 複製一份
        render_kwargs['features'] = z

        # ========== 渲染 ==========
        # 評估模式使用原始方法（更穩定），訓練模式使用 NerfAcc（更快）
        if self.use_nerfacc and not self.use_test_kwargs:
            # NerfAcc 渲染 - 只在訓練時使用
            rgb, disp, acc, extras = render_nerfacc(
                self.H, self.W, self.focal, label,
                rays=rays,
                near=self.near, 
                far=self.far,
                use_viewdirs=render_kwargs.get('use_viewdirs', True),
                estimator=self.estimator,
                render_step_size=self.render_step_size,
                # 只傳遞 render_nerfacc 需要的參數
                network_fn=render_kwargs['network_fn'],
                network_query_fn=render_kwargs['network_query_fn'],
                features=render_kwargs.get('features'),
                network_fine=render_kwargs.get('network_fine'),
            )
        else:
            # 原本的渲染 - 評估時使用
            rgb, disp, acc, extras = render(
                self.H, self.W, self.focal, label,
                chunk=self.chunk, rays=rays,
                **render_kwargs
            )

        rays_to_output = lambda x: x.view(len(x), -1) * 2 - 1
    
        if self.use_test_kwargs:
            return rays_to_output(rgb), \
                   rays_to_output(disp), \
                   rays_to_output(acc), extras

        rgb = rays_to_output(rgb)

        # CCSR 處理
        ccsr_output = None
        if self.use_ccsr and return_ccsr_output:
            total_elements = rgb.numel()
            rgb_nerf = rgb.view(bs, total_elements // (bs * 3), 3)
            nerf_images = rgb_nerf.view(bs, int(np.sqrt(rgb_nerf.shape[1])), int(np.sqrt(rgb_nerf.shape[1])), 3).permute(0, 3, 1, 2)
            
            patch_size = 64
            lr_size = max(8, patch_size // 4)
            lr_images = F.interpolate(nerf_images, size=(lr_size, lr_size), mode='bilinear', align_corners=False)
            
            ccsr_results = []
            for i in range(bs):
                angle_idx = int(label[i, 2].item())
                view_idx = (angle_idx * 8) // 360
                ccsr_result = self.ccsr(lr_images[i:i+1], view_idx)
                ccsr_results.append(ccsr_result)
            
            ccsr_combined = torch.cat(ccsr_results, dim=0)
            ccsr_resized = F.interpolate(ccsr_combined, size=(patch_size, patch_size), 
                                        mode='bilinear', align_corners=False)

        if return_ccsr_output:
            ccsr_output = ccsr_resized.permute(0, 2, 3, 1).contiguous().view(-1, 3)
            return rgb, rays, ccsr_output
        else:
            return rgb, rays

    def update_occupancy_grid(self, step):
        """
        更新 occupancy grid（簡單版本）
        使用球形估計
        """
        if not self.use_nerfacc or self.estimator is None:
            return
            
        def occ_eval_fn(positions):
            """基於球形的簡單佔據估計"""
            dist = torch.norm(positions, dim=-1)
            density = torch.where(
                dist < self.radius,
                torch.ones_like(dist),
                torch.zeros_like(dist)
            )
            return density
        
        self.estimator.update_every_n_steps(
            step=step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=1e-2,
        )

    def update_occupancy_grid_with_network(self, step, label, z_sample=None):
        """
        使用 NeRF 網路更新 occupancy grid（精確版本）
        改進：使用多個隨機view directions的平均密度
        """
        if not self.use_nerfacc or self.estimator is None:
            return

        network_fn = self.render_kwargs_train['network_fn']
        network_query_fn = self.render_kwargs_train['network_query_fn']

        def occ_eval_fn(positions):
            """使用 NeRF 網路評估密度（多view平均）"""
            with torch.no_grad():
                chunk_size = 32768  # 減小chunk避免OOM
                n_views = 4  # 使用4個隨機view方向

                # 生成多個隨機view directions
                random_dirs = torch.randn(n_views, 3, device=positions.device)
                random_dirs = F.normalize(random_dirs, dim=-1)

                all_sigmas = []

                # 對每個view direction評估密度
                for view_dir in random_dirs:
                    sigmas = []
                    for i in range(0, positions.shape[0], chunk_size):
                        pos_chunk = positions[i:i+chunk_size]
                        n_pos = pos_chunk.shape[0]

                        # 為這批位置使用相同的view direction
                        view_chunk = view_dir.unsqueeze(0).expand(n_pos, 3)

                        raw = network_query_fn(
                            pos_chunk.unsqueeze(0),
                            view_chunk,
                            network_fn,
                            label,
                            z_sample
                        )
                        sigma = torch.relu(raw[0, :, 3])
                        sigmas.append(sigma)

                    all_sigmas.append(torch.cat(sigmas, dim=0))

                # 取多個view的平均密度（更robust）
                avg_sigmas = torch.stack(all_sigmas, dim=0).mean(dim=0)

            return avg_sigmas

        self.estimator.update_every_n_steps(
            step=step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=1e-2,
        )

    def decrease_nerf_noise(self, it):
        end_it = 5000
        if it < end_it:
            noise_std = self.initial_raw_noise_std - self.initial_raw_noise_std/end_it * it
            self.render_kwargs_train['raw_noise_std'] = noise_std

    def sample_pose(self):
        loc = sample_on_sphere(self.range_u, self.range_v)
        radius = self.radius
        if isinstance(radius, tuple):
            radius = np.random.uniform(*radius)

        loc = loc * radius
        R = look_at(loc)[0]

        RT = np.concatenate([R, loc.reshape(3, 1)], axis=1)
        RT = torch.Tensor(RT.astype(np.float32))
        return RT

    def sample_select_pose(self, u, v):
        radius = self.radius
        loc = to_sphere(u, v) * radius
        R = look_at(loc)[0]
        
        RT = np.concatenate([R, loc.reshape(3, 1)], axis=1)
        RT = torch.Tensor(RT.astype(np.float32))
        
        return RT
  
    def sample_rays(self):
        pose = self.sample_pose()
        sampler = self.val_ray_sampler if self.use_test_kwargs else self.ray_sampler 
        batch_rays, _, _ = sampler(self.H, self.W, self.focal, pose)
        return batch_rays
    
    def sample_select_rays(self, u, v):
        pose = self.sample_select_pose(u, v)
        sampler = self.val_ray_sampler if self.use_test_kwargs else self.ray_sampler
        batch_rays, _, _ = sampler(self.H, self.W, self.focal, pose)
        return batch_rays

    def to(self, device):
        self.render_kwargs_train['network_fn'].to(device)
        if self.use_nerfacc and self.estimator is not None:
            self.estimator.to(device)
        self.device = device
        return self

    def train(self):
        self.use_test_kwargs = False
        self.render_kwargs_train['network_fn'].train()

    def eval(self):
        self.use_test_kwargs = True
        self.render_kwargs_train['network_fn'].eval()