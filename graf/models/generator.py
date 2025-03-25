import numpy as np
import torch
from ..utils import sample_on_sphere, look_at, to_sphere
from graf.transforms import ImgToPatch
from ..transforms import FullRaySampler
from submodules.nerf_pytorch.run_nerf_mod import render, run_network            # import conditional render
from functools import partial
import torch.nn.functional as F  
from graf.models.vit_model import ViewConsistencyTransformer
import os
import pickle


class Generator(object):
    def __init__(self, H, W, focal, radius, ray_sampler, render_kwargs_train, render_kwargs_test, parameters, named_parameters,
                 range_u=(0,1), range_v=(0.01,0.49),v=0, chunk=None, device='cuda', orthographic=False, use_default_rays=False, reference_images=None, initial_direction=None, use_vit=True):
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
        
        for k, v in self.module_dict.items():
            if k in ['generator']:
                continue       # parameters already included
            self._parameters += list(v.parameters())
            self._named_parameters += list(v.named_parameters())
        
        self.parameters = lambda: self._parameters           # save as function to enable calling model.parameters()
        self.named_parameters = lambda: self._named_parameters           # save as function to enable calling model.named_parameters()
        self.use_test_kwargs = False

        self.render = partial(render, H=self.H, W=self.W, focal=self.focal, chunk=self.chunk)

        # 添加 ViT 視角一致性模型
        self.use_vit = use_vit
        self.cached_views = None

        if use_vit:
            self.view_transformer = ViewConsistencyTransformer(
                img_size=32,
                patch_size=4,
                in_chans=3,
                embed_dim=256
            ).to(device)

            # 添加到模塊字典中以便管理參數
            self.module_dict['view_transformer'] = self.view_transformer
            self._parameters += list(self.view_transformer.parameters())
            self._named_parameters += list(self.view_transformer.named_parameters())

    # 添加一個方法來更新緩存的視圖
    def update_cached_views(self, views):
        """更新用於 ViT 分析的緩存視圖"""
        self.cached_views = views.to(self.device)
    
    # 添加一個幫助函數來預訓練 ViT
    def pretrain_view_transformer(self, dataloader, epochs=5, lr=1e-4):
        """預訓練視角變換器，使其能夠預測準確的角度"""
        if not self.use_vit:
            print("ViT 未啟用，無法預訓練")
            return
        
        optimizer = torch.optim.Adam(self.view_transformer.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0
            for imgs, labels in dataloader:
                optimizer.zero_grad()
                
                imgs = imgs.to(self.device)
                hwf = [128, 128, 288.68534423437166]
                img_to_patch = ImgToPatch(self.ray_sampler, hwf)
                rgbs = img_to_patch(imgs)
                
                # 獲取角度標籤
                h_angles = labels[:, 2].float().to(self.device) / 360.0  # 水平角度，規範化到 0-1
                
                # 假設垂直角度在標籤的第四列，否則使用默認值 0.5
                if labels.shape[1] > 3:
                    v_angles = labels[:, 3].float().to(self.device)  # 假設已經在 0-0.5 範圍內
                else:
                    v_angles = torch.ones_like(h_angles) * 0.25  # 默認垂直角度
                
                # 組合為目標角度張量
                target_angles = torch.stack([h_angles, v_angles], dim=1)
                
                # 預測角度
                pred_angles = self.view_transformer(rgbs)
                
                # 計算損失
                loss = F.mse_loss(pred_angles, target_angles)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss/len(dataloader):.6f}")
        
        print("視角變換器預訓練完成")

    def __call__(self, z, label, rays=None):
        bs = z.shape[0]
        if rays is None:
            if self.use_default_rays :
                rays = torch.cat([self.sample_rays() for _ in range(bs)], dim=1)
            else:
                all_rays = []
                v_list = [float(x.strip()) for x in self.v.split(",")]

                # 如果使用 ViT 且有緩存視圖，預測角度
                angles = None
                if self.use_vit and self.cached_views is not None:
                    angles = self.view_transformer(self.cached_views)

                for i in range(label.size(0)):
                    second_value = label[i, 1].item()
                    index = int(label[i, 2].item())  # 得到第3個值
                    
                    # 基礎 u 值計算
                    selected_u = index / 360
                    
                    # 如果使用 ViT 且有預測角度，應用預測結果
                    if self.use_vit and angles is not None:
                        # 使用預測的水平角度（或混合使用）
                        pred_u = angles[i, 0].item()
                        # 可以選擇直接使用預測值或與原始值混合
                        selected_u = (selected_u + pred_u) / 2  # 混合原始值和預測值
                        
                        # 使用預測的垂直角度
                        selected_v = angles[i, 1].item()  # 已在 0-0.5 範圍內
                    else:
                        # 使用預設的垂直角度
                        selected_v = v_list[min(int(second_value), len(v_list) - 1)]
                    
                    # 使用選定的角度生成光線
                    rays = torch.cat([self.sample_select_rays(selected_u, selected_v)], dim=1)
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
        return rgb, rays

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

    