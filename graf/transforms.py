import torch
from math import sqrt, exp

from submodules.nerf_pytorch.run_nerf_helpers_mod import get_rays, get_rays_ortho


class ImgToPatch(object):
    def __init__(self, ray_sampler, hwf):
        self.ray_sampler = ray_sampler
        self.hwf = hwf      # camera intrinsics

    def __call__(self, img):
        rgbs = []
        for img_i in img:
            pose = torch.eye(4)         # use dummy pose to infer pixel values
            _, selected_idcs, pixels_i = self.ray_sampler(H=self.hwf[0], W=self.hwf[1], focal=self.hwf[2], pose=pose)
            if selected_idcs is not None:
                rgbs_i = img_i.flatten(1, 2).t()[selected_idcs]
            else:
                rgbs_i = torch.nn.functional.grid_sample(img_i.unsqueeze(0), 
                                     pixels_i.unsqueeze(0), mode='bilinear', align_corners=True)[0]
                rgbs_i = rgbs_i.flatten(1, 2).t()
            rgbs.append(rgbs_i)

        rgbs = torch.cat(rgbs, dim=0)       # (B*N)x3

        return rgbs


class RaySampler(object): #生成相機射線
    def __init__(self, N_samples, orthographic=False):
        super(RaySampler, self).__init__()
        self.N_samples = N_samples
        self.scale = torch.ones(1,).float()
        self.return_indices = True
        self.orthographic = orthographic

    def __call__(self, H, W, focal, pose):
        if self.orthographic: #正射投影
            size_h, size_w = focal      # Hacky
            rays_o, rays_d = get_rays_ortho(H, W, pose, size_h, size_w)
        else: #透射投影
            rays_o, rays_d = get_rays(H, W, focal, pose)

        select_inds = self.sample_rays(H, W) #取樣射線

        if self.return_indices:
            rays_o = rays_o.view(-1, 3)[select_inds] #光線原點
            rays_d = rays_d.view(-1, 3)[select_inds] #光線方向

            h = (select_inds // W) / float(H) - 0.5
            w = (select_inds %  W) / float(W) - 0.5

            hw = torch.stack([h,w]).t()

        else:
            rays_o = torch.nn.functional.grid_sample(rays_o.permute(2,0,1).unsqueeze(0), 
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            rays_d = torch.nn.functional.grid_sample(rays_d.permute(2,0,1).unsqueeze(0), 
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            rays_o = rays_o.permute(1,2,0).view(-1, 3)
            rays_d = rays_d.permute(1,2,0).view(-1, 3)

            hw = select_inds
            select_inds = None

        return torch.stack([rays_o, rays_d]), select_inds, hw

    def sample_rays(self, H, W):
        raise NotImplementedError


class FullRaySampler(RaySampler):
    def __init__(self, **kwargs):
        super(FullRaySampler, self).__init__(N_samples=None, **kwargs)

    def sample_rays(self, H, W):
        return torch.arange(0, H*W) #生成了一個從 0 到 H*W-1 的長度為 H*W 的一維張量



class FlexGridRaySampler(RaySampler):
    def __init__(self, N_samples, random_shift=True, random_scale=True, min_scale=0.25, max_scale=1., scale_anneal=-1,
                 **kwargs):
        self.N_samples_sqrt = int(sqrt(N_samples))
        super(FlexGridRaySampler, self).__init__(self.N_samples_sqrt**2, **kwargs)

        self.random_shift = random_shift
        self.random_scale = random_scale

        self.min_scale = min_scale
        self.max_scale = max_scale

        # nn.functional.grid_sample grid value range in [-1,1]
        self.w, self.h = torch.meshgrid([torch.linspace(-1,1,self.N_samples_sqrt),
                                         torch.linspace(-1,1,self.N_samples_sqrt)])
        self.h = self.h.unsqueeze(2)
        self.w = self.w.unsqueeze(2)

        # directly return grid for grid_sample
        self.return_indices = False

        self.iterations = 0
        self.scale_anneal = scale_anneal

    def sample_rays(self, H, W):

        h = self.h.clone()
        w = self.w.clone()

        return torch.cat([h, w], dim=2) #torch.Size([32, 32, 2])


class BottomFocusedRaySampler(RaySampler):
    """
    专注于图片下半部分的光线采样器
    在下半部分采样更多的点，以提升下半部分的细节
    """
    def __init__(self, N_samples, bottom_ratio=0.7, **kwargs):
        """
        Args:
            N_samples: 总采样数量
            bottom_ratio: 下半部分的采样比例 (0.7 表示70%的点在下半部分)
        """
        super(BottomFocusedRaySampler, self).__init__(N_samples, **kwargs)
        self.bottom_ratio = bottom_ratio

    def sample_rays(self, H, W):
        """
        采样光线，让更多的采样点集中在图片下半部分

        Args:
            H: 图片高度
            W: 图片宽度

        Returns:
            采样点的索引 (1D tensor)
        """
        N_bottom = int(self.N_samples * self.bottom_ratio)
        N_top = self.N_samples - N_bottom

        # 下半部分采样 (从 H/2 到 H)
        bottom_h = torch.randint(H // 2, H, (N_bottom,))
        bottom_w = torch.randint(0, W, (N_bottom,))
        bottom_indices = bottom_h * W + bottom_w

        # 上半部分采样 (从 0 到 H/2)
        top_h = torch.randint(0, H // 2, (N_top,))
        top_w = torch.randint(0, W, (N_top,))
        top_indices = top_h * W + top_w

        # 合并并打乱索引
        # 重要：必须打乱！因为判别器会把这些点重塑成固定形状的patch
        # 如果不打乱，"前70%总是下半部分"会成为判别器可利用的虚假模式
        all_indices = torch.cat([bottom_indices, top_indices])
        perm = torch.randperm(all_indices.size(0))

        return all_indices[perm]
