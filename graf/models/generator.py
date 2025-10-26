import numpy as np
import torch
from ..utils import sample_on_sphere, look_at, to_sphere
from graf.transforms import ImgToPatch
from ..transforms import FullRaySampler
from submodules.nerf_pytorch.run_nerf_mod import render, run_network            # import conditional render
from functools import partial
import torch.nn.functional as F
from graf.models.ccsr import CCSR
from graf.models.esrgan_model import ESRGANWrapper
import os
import pickle


class Generator(object):
    def __init__(self, H, W, focal, radius, ray_sampler, render_kwargs_train, render_kwargs_test, parameters, named_parameters,
                 range_u=(0,1), range_v=(0.01,0.49),v=0, chunk=None, device='cuda', orthographic=False, use_default_rays=False,
                 use_ccsr=True, num_views=8, use_esrgan=False, esrgan_pretrained_path=None, esrgan_freeze=True):
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
        self.use_esrgan = use_esrgan

        coords = torch.from_numpy(np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), -1))
        self.coords = coords.view(-1, 2)

        self.ray_sampler = ray_sampler   #FlexGridRaySampler
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

        # 添加超分辨率模組
        if self.use_esrgan:
            # 使用 ESRGAN 預訓練模型
            print(f"使用 ESRGAN 模型，預訓練路徑: {esrgan_pretrained_path}")
            self.esrgan = ESRGANWrapper(
                pretrained_path=esrgan_pretrained_path,
                freeze=esrgan_freeze,
                scale=4
            ).to(device)
            self.module_dict['esrgan'] = self.esrgan

        elif self.use_ccsr:
            # 使用自定義 CCSR 模組
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
        self.render = partial(render, H=self.H, W=self.W, focal=self.focal, chunk=self.chunk)

    def __call__(self, z, label, rays=None, return_sr_output=False):
        bs = z.shape[0]
        if rays is None:
            if self.use_default_rays :
                rays = torch.cat([self.sample_rays() for _ in range(bs)], dim=1)
            else:
                all_rays = []
                v_list = [float(x.strip()) for x in self.v.split(",")]

                for i in range(label.size(0)):
                    second_value = label[i, 1].item()
                    index = int(label[i, 2].item())  # 得到第3個值

                    # 基礎 u v值計算
                    selected_u = index / 360
                    selected_v = v_list[int(second_value)]

                    # 使用選定的角度生成光線
                    rays = self.sample_select_rays(selected_u, selected_v)
                    all_rays.append(rays)

                rays = torch.cat(all_rays, dim=1)


        render_kwargs = self.render_kwargs_test if self.use_test_kwargs else self.render_kwargs_train
        render_kwargs = dict(render_kwargs)        # copy

        render_kwargs['features'] = z
        rgb, disp, acc, extras = render(self.H, self.W, self.focal, label, chunk=self.chunk, rays=rays,
                                        **render_kwargs)

        rays_to_output = lambda x: x.view(len(x), -1) * 2 - 1      # (BxN_samples)xC

        if self.use_test_kwargs:               # return all outputs
            return rays_to_output(rgb), \
                   rays_to_output(disp), \
                   rays_to_output(acc), extras

        rgb_nerf = rays_to_output(rgb)

        # 超分辨率處理
        sr_output = None
        if (self.use_esrgan or self.use_ccsr) and return_sr_output:
            # 將 NeRF 輸出轉換為圖像格式
            total_elements = rgb_nerf.numel()
            rgb_nerf_reshaped = rgb_nerf.view(bs, total_elements // (bs * 3), 3)
            patch_size = int(np.sqrt(rgb_nerf_reshaped.shape[1]))
            nerf_images = rgb_nerf_reshaped.view(bs, patch_size, patch_size, 3).permute(0, 3, 1, 2)

            if self.use_esrgan:
                # 使用 ESRGAN 進行超分辨率
                # NeRF 渲染的是 patch_size (64x64)，先下採樣到 1/4 再用 ESRGAN 上採樣
                lr_size = patch_size // 4  # 16x16
                lr_images = F.interpolate(nerf_images, size=(lr_size, lr_size),
                                        mode='bilinear', align_corners=False)

                # ESRGAN 超分辨率 (16x16 -> 64x64)
                sr_images = self.esrgan(lr_images)

                # 調整到目標尺寸
                sr_resized = F.interpolate(sr_images, size=(patch_size, patch_size),
                                         mode='bilinear', align_corners=False)

            elif self.use_ccsr:
                # 使用 CCSR 進行超分辨率
                lr_size = max(8, patch_size // 4)
                lr_images = F.interpolate(nerf_images, size=(lr_size, lr_size),
                                        mode='bilinear', align_corners=False)

                # 對每個樣本應用 CCSR
                ccsr_results = []
                for i in range(bs):
                    angle_idx = int(label[i, 2].item())
                    view_idx = (angle_idx * 8) // 360  # 映射到 0-7
                    ccsr_result = self.ccsr(lr_images[i:i+1], view_idx)
                    ccsr_results.append(ccsr_result)

                sr_combined = torch.cat(ccsr_results, dim=0)
                sr_resized = F.interpolate(sr_combined, size=(patch_size, patch_size),
                                         mode='bilinear', align_corners=False)

            # 轉換回與 NeRF 相同的格式
            sr_output = sr_resized.permute(0, 2, 3, 1).contiguous().view(-1, 3)

        if return_sr_output:
            return rgb_nerf, rays, sr_output
        else:
            return rgb_nerf, rays

    def decrease_nerf_noise(self, it):
        end_it = 5000
        if it < end_it:
            noise_std = self.initial_raw_noise_std - self.initial_raw_noise_std/end_it * it
            self.render_kwargs_train['raw_noise_std'] = noise_std

    def sample_pose(self):   #計算旋轉矩陣(相機姿勢)  train
        # sample location on unit sphere
        #print("Type of self.v:", type(self.v))
        loc = sample_on_sphere(self.range_u, self.range_v)
        # loc = to_sphere(u, v)
        
        # sample radius if necessary
        radius = self.radius
        if isinstance(radius, tuple):
            radius = np.random.uniform(*radius)

        loc = loc * radius
        R = look_at(loc)[0]

        RT = np.concatenate([R, loc.reshape(3, 1)], axis=1)
        RT = torch.Tensor(RT.astype(np.float32))
        return RT


    def sample_select_pose(self, u, v):   #計算旋轉矩陣(相機姿勢)
        # sample location on unit sphere
        #print("Type of self.v:", type(self.v))
        
        # sample radius if necessary
        radius = self.radius
        
        # 正常的球面取樣
        loc = to_sphere(u, v) * radius
        R = look_at(loc)[0]
        
        RT = np.concatenate([R, loc.reshape(3, 1)], axis=1)
        RT = torch.Tensor(RT.astype(np.float32))
        
        return RT
  
    def sample_rays(self):   #設train用的rays
        pose = self.sample_pose()
        # print(f"`trainpose`:{pose}")
        sampler = self.val_ray_sampler if self.use_test_kwargs else self.ray_sampler 
        batch_rays, _, _ = sampler(self.H, self.W, self.focal, pose)
        return batch_rays #torch.Size([2, 1024, 3])
    
    def sample_select_rays(self, u ,v):
        pose = self.sample_select_pose(u, v)
        #print(f"trainpose:{pose}")
        sampler = self.val_ray_sampler if self.use_test_kwargs else self.ray_sampler  #如果 self.use_test_kwargs 為真，則使用 self.val_ray_sampler
        batch_rays, _, _ = sampler(self.H, self.W, self.focal, pose)
        return batch_rays

    def to(self, device):
        self.render_kwargs_train['network_fn'].to(device)
        self.device = device
        return self

    def train(self):
        self.use_test_kwargs = False
        self.render_kwargs_train['network_fn'].train()

    def eval(self):
        self.use_test_kwargs = True
        self.render_kwargs_train['network_fn'].eval()

    