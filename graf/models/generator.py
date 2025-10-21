import numpy as np
import torch
from ..utils import sample_on_sphere, look_at, to_sphere 
from graf.transforms import ImgToPatch
from ..transforms import FullRaySampler
from submodules.nerf_pytorch.run_nerf_mod import render, run_network            # import conditional render
from functools import partial
import torch.nn.functional as F  
from graf.models.ccsr import CCSR
import os
import pickle


class Generator(object):
    def __init__(self, H, W, focal, radius, ray_sampler, render_kwargs_train, render_kwargs_test, parameters, named_parameters,
                 range_u=(0,1), range_v=(0.01,0.49),v=0, chunk=None, device='cuda', orthographic=False, use_default_rays=False, use_ccsr=True, num_views=8):
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

        # 添加CCSR模組
        if self.use_ccsr:
            # 假設低分辨率圖像尺寸為原圖的1/4
            lr_height, lr_width = H // 4, W // 4
            self.ccsr = CCSR(num_views, lr_height, lr_width, scale_factor=4).to(device)
            
            # 將CCSR參數加入到優化器參數列表中
            self._parameters = parameters + list(self.ccsr.parameters())
            self._named_parameters = named_parameters + list(self.ccsr.named_parameters())
        else:
            self._parameters = parameters
            self._named_parameters = named_parameters

        coords = torch.from_numpy(np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), -1))
        self.coords = coords.view(-1, 2)
        self.initial_direction = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

        self.ray_sampler = ray_sampler   #FlexGridRaySampler
        self.val_ray_sampler = FullRaySampler(orthographic=orthographic)
        self.render_kwargs_train = render_kwargs_train
        self.render_kwargs_test = render_kwargs_test
        self.initial_raw_noise_std = self.render_kwargs_train['raw_noise_std']
        self._parameters = parameters
        self._named_parameters = named_parameters
        self.module_dict = {'generator': self.render_kwargs_train['network_fn']}

        # 添加CCSR到module_dict
        if self.use_ccsr:
            self.module_dict['ccsr'] = self.ccsr
        
        for k, v in self.module_dict.items():
            if k in ['generator']:
                continue       # parameters already included
            self._parameters += list(v.parameters())
            self._named_parameters += list(v.named_parameters())
        
        self.parameters = lambda: self._parameters           # save as function to enable calling model.parameters()
        self.named_parameters = lambda: self._named_parameters           # save as function to enable calling model.named_parameters()
        self.use_test_kwargs = False

        self.render = partial(render, H=self.H, W=self.W, focal=self.focal, chunk=self.chunk)

    def __call__(self, z, label, rays=None, return_ccsr_output=False):
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

        rgb = rays_to_output(rgb)

        # 如果啟用CCSR並且需要返回CCSR輸出
        if self.use_ccsr and return_ccsr_output:
            with torch.no_grad():  # CCSR 處理時暫時不需要梯度
                # 將NeRF輸出轉換為圖像格式進行CCSR處理
                total_elements = rgb.numel()
                rgb_nerf = rgb.view(bs, total_elements // (bs * 3), 3)

                # 驗證尺寸是否為完美平方數
                num_pixels = rgb_nerf.shape[1]
                patch_size = int(np.sqrt(num_pixels))
                assert patch_size * patch_size == num_pixels, f"patch採樣數量 {num_pixels} 不是完美平方數"

                nerf_images = rgb_nerf.view(bs, patch_size, patch_size, 3).permute(0, 3, 1, 2)

                # 生成低分辨率版本
                lr_size = max(8, patch_size // 4)
                lr_images = F.interpolate(nerf_images, size=(lr_size, lr_size),
                                         mode='bilinear', align_corners=False)

                # 對每個樣本應用CCSR（批次處理以節省記憶體）
                ccsr_results = []
                for i in range(bs):
                    # 從 label 提取正確的視角索引
                    if label.shape[1] > 2:
                        angle_idx = int(label[i, 2].item())  # 0-359
                        view_idx = (angle_idx * self.ccsr.cclc.num_views) // 360  # 映射到 0-7
                    else:
                        view_idx = i % self.ccsr.cclc.num_views

                    ccsr_result = self.ccsr(lr_images[i:i+1], view_idx)
                    ccsr_results.append(ccsr_result)

                ccsr_combined = torch.cat(ccsr_results, dim=0)
                del ccsr_results, lr_images  # 釋放中間結果

                ccsr_resized = F.interpolate(ccsr_combined, size=(patch_size, patch_size),
                                           mode='bilinear', align_corners=False)
                del ccsr_combined  # 釋放

            # 重新啟用梯度並轉換格式
            ccsr_output = ccsr_resized.permute(0, 2, 3, 1).contiguous().view(-1, 3)
            ccsr_output.requires_grad_(True)  # 確保可以計算梯度

            return rgb, rays, ccsr_output
        else:
            return rgb, rays
        
        # return rgb, rays

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
    
    def get_canonical_poses(self):
        """
        返回固定世界座標上的一組標準相機姿勢。
        這些作為世界座標系統的參考點。
        
        返回：
            標準姿勢的字典
        """
        canonical_poses = {
            "front": self.sample_select_pose(0.0, 0.5),    # 0°（前）
            "right": self.sample_select_pose(0.25, 0.5),   # 90°（右）
            "back": self.sample_select_pose(0.5, 0.5),     # 180°（後）
            "left": self.sample_select_pose(0.75, 0.5),    # 270°（左）
            "top": self.sample_select_pose(0.0, 0.0),      # 頂視圖
            "bottom": self.sample_select_pose(0.0, 1.0)    # 底視圖
        }
        return canonical_poses
    
    def initialize_world_coordinates(self):
        """
        初始化並驗證固定的世界座標系統。
        應該在訓練開始時調用一次。
        """
        # 獲取標準視圖
        canonical_poses = self.get_canonical_poses()
        
        # 確保原點在 (0,0,0)
        origin = torch.zeros(3, device=self.device)
        
        # 在場景中創建視覺標記來表示世界軸
        # 這僅用於調試/可視化目的
        self.world_axes = {
            "x": torch.tensor([1.0, 0.0, 0.0], device=self.device),
            "y": torch.tensor([0.0, 1.0, 0.0], device=self.device),
            "z": torch.tensor([0.0, 0.0, 1.0], device=self.device)
        }
        
        print("世界座標系統已初始化。")
        print(f"原點: {origin}")
        print(f"前視圖位置: {canonical_poses['front'][:3, 3]}")
        
        return canonical_poses

    
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

    