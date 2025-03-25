import argparse
import os
import time
import torch.nn.functional as F
import random
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import wandb
import sys
sys.path.append('submodules')

from graf.gan_training import Evaluator
from graf.config import get_data, build_models, load_config, save_config
from graf.utils import get_zdist, visualize_coordinate_system
from graf.train_step import compute_grad2, compute_loss, save_data, wgan_gp_reg
from graf.transforms import ImgToPatch
from graf.models.vit_model import ViewConsistencyTransformer
 
from GAN_stability.gan_training.checkpoints_mod import CheckpointIO

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def setup_directories(config):
    out_dir = os.path.join(config['training']['outdir'], config['expname'])
    checkpoint_dir = os.path.join(out_dir, 'chkpts')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return out_dir, checkpoint_dir

def initialize_training(config, device):
    # dataset
    train_dataset, hwfr= get_data(config)
    if config['data']['orthographic']:
        hw_ortho = (config['data']['far']-config['data']['near'],) * 2
        hwfr[2] = hw_ortho
    config['data']['hwfr'] = hwfr
    
    # train_loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['nworkers'],
        shuffle=True, 
        pin_memory=True,
        sampler=None, 
        drop_last=True,
        generator=torch.Generator(device='cuda:0')
    )
    
    # Create models
    generator, discriminator = build_models(config)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    return train_loader, generator, discriminator

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 確定性算法
    torch.backends.cudnn.benchmark = False  # 關閉自動優化

def main():
    set_random_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()

    # load config
    config = load_config(args.config)
    config['data']['fov'] = float(config['data']['fov'])
    restart_every = config['training']['restart_every']
    batch_size=config['training']['batch_size']
    fid_every = config['training']['fid_every']
    device = torch.device("cuda:0")
    
    # 創建目錄
    out_dir, checkpoint_dir = setup_directories(config)
    save_config(os.path.join(out_dir, 'config.yaml'), config)
    
    # 初始化model
    train_loader, generator, discriminator = initialize_training(config, device)

    # 初始化世界座標系統
    canonical_poses = generator.initialize_world_coordinates()
    canonical_pose_path = os.path.join(out_dir, 'canonical_poses.pt')
    torch.save(canonical_poses, canonical_pose_path)
    print(f"標準姿勢已保存到 {canonical_pose_path}")
    
    coordinate_viz_path = visualize_coordinate_system(generator, out_dir, it=0)
    print(f"座標系統可視化已保存到 {coordinate_viz_path}")

    # 優化器
    lr_g = config['training']['lr_g']
    lr_d = config['training']['lr_d']
    g_params = generator.parameters()
    d_params = discriminator.parameters()
    g_optimizer = optim.RMSprop(g_params, lr=lr_g, alpha=0.99, eps=1e-8)
    d_optimizer = optim.RMSprop(d_params, lr=lr_d, alpha=0.99, eps=1e-8)

    #get patch
    hwfr = config['data']['hwfr']
    img_to_patch = ImgToPatch(generator.ray_sampler, hwfr[:3])
    
    # 初始化 wandb
    wandb.init(
        project="graf250311",
        entity="vicky20020808",
        name="RS615",
        config=config
    )

    # 如果啟用了 ViT，預訓練視角變換器
    if hasattr(generator, 'use_vit') and generator.use_vit:
        print("開始預訓練視角一致性模型...")
        # 可以創建一個專用於預訓練的數據加載器，或者使用現有的
        generator.pretrain_view_transformer(train_loader, epochs=5)
    
    # 設置檢查點
    checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)
    checkpoint_io.register_modules(
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        **generator.module_dict
    )
    
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)

    # Evaluator
    evaluator = Evaluator(fid_every > 0, generator, zdist, None,
                          batch_size=batch_size, device=device, inception_nsamples=33)
    
    it = epoch_idx = -1
    tstart = t0 = time.time()
    
    while True:
        epoch_idx += 1
        for x_real, label in tqdm(train_loader, desc=f"Epoch {epoch_idx}"):
            it += 1
            
            generator.ray_sampler.iterations = it
            generator.train()
            discriminator.train()

            # Discriminator updates
            d_optimizer.zero_grad()

            x_real = x_real.to(device)
            rgbs = img_to_patch(x_real)
            rgbs.requires_grad_(True)

            # 如果使用 ViT，先更新緩存的視圖
            if hasattr(generator, 'use_vit') and generator.use_vit:
                # 將真實圖像轉換為合適的格式
                real_views = rgbs.to(device)
                generator.update_cached_views(real_views)

            z = zdist.sample((batch_size,))
            x_fake, rays = generator(z, label)

            d_real = discriminator(rgbs, label)
            dloss_real = compute_loss(d_real, 1)
            reg = 10. * compute_grad2(d_real, rgbs).mean()
            

            d_fake = discriminator(x_fake, label)
            dloss_fake = compute_loss(d_fake, 0)
            # reg = 10. * wgan_gp_reg(discriminator, rgbs, x_fake, label)

            dloss = dloss_real + dloss_fake
            dloss_all = dloss_real + dloss_fake +reg
            dloss_all.backward()
            d_optimizer.step()

            # Generators updates
            if config['nerf']['decrease_noise']:
                generator.decrease_nerf_noise(it)

            g_optimizer.zero_grad()

            z = zdist.sample((batch_size,))
            x_fake, rays= generator(z, label)
            d_fake = discriminator(x_fake, label)

            gloss_disc = compute_loss(d_fake, 1) 

            # 添加視角一致性損失
            gloss_view = 0
            if hasattr(generator, 'use_vit') and generator.use_vit:
                # 將生成的圖像添加到緩存
                fake_views = x_fake.view(batch_size, 32, 32, 3).permute(0, 3, 1, 2)
                
                # 獲取角度標籤
                h_angles = label[:, 2].float() / 360.0  # 水平角度，規範化到 0-1
                
                # 根据 label[:,1] 的值选择对应的垂直角度
                v_angles = torch.zeros_like(h_angles)
                
                # 垂直角度映射表
                v_angle_map = {
                    0: 0.5,
                    1: 0.4166667,
                    2: 0.3333334,
                    3: 0.25,
                    4: 0.1666667
                }
                
                # 根据标签设置垂直角度
                for i, v_idx in enumerate(label[:, 1].long()):
                    v_angles[i] = v_angle_map.get(v_idx.item(), 0.25)
                
                # 組合為目標角度張量
                target_angles = torch.stack([h_angles, v_angles], dim=1).to(device)
                
                # 預測角度
                pred_angles = generator.view_transformer(fake_views)
                
                # 計算視角一致性損失
                gloss_view = F.mse_loss(pred_angles, target_angles)
                
                # 添加到總損失，使用權重係數
                gloss = gloss_disc + 0.1 * gloss_view
            else:
                gloss = gloss_disc

            gloss.backward()
            g_optimizer.step()
                
            # wandb
            if (it + 1) % config['training']['print_every'] == 0:
                wandb.log({
                    "loss/discriminator": dloss,
                    "loss/generator": gloss_disc,
                    "loss/view_consistency": gloss_view if hasattr(generator, 'use_vit') and generator.use_vit else 0,
                    "loss/regularizer": reg,
                    "iteration": it
                })
            
            # 在需要儲存資料的位置，例如在訓練迴圈中特定迭代次數時
            if (it % 6000 == 0):
                # 在這裡使用當前的 label 和 rays
                save_data(label, rays, it, save_dir=os.path.join(out_dir, 'saved_data'))

            # (ii) Sample if necessary
            if ((it % config['training']['sample_every']) == 0) or ((it < 500) and (it % 100 == 0)):
                # is_training = generator.use_test_kwargs
                # generator.eval()  
                plist = []
                angle_positions = [(i/8, 0.5) for i in range(8)] 
                ztest = zdist.sample((batch_size,))
                label_test = torch.tensor([[0] if i < 4 else [0] for i in range(batch_size)])

                # save_dir = os.path.join(out_dir, 'poses')
                # os.makedirs(save_dir, exist_ok=True)

                for i, (u, v) in enumerate(angle_positions):
                    # print(f"指定角度:{u}, 轉換後角度:{position_angle}")
                    poses = generator.sample_select_pose(u ,v)
                    plist.append(poses)
                ptest = torch.stack(plist)

                rgb, depth, acc = evaluator.create_samples(ztest.to(device), label_test, ptest)

                # 添加視角一致性可視化
                if hasattr(generator, 'use_vit') and generator.use_vit and generator.cached_views is not None:
                    # 獲取當前的預測角度
                    with torch.no_grad():
                        pred_angles = generator.view_transformer(generator.cached_views)
                    
                    # 記錄預測的角度
                    for i in range(batch_size):  # 最多記錄4個樣本
                        wandb.log({
                            f"angles/sample_{i}_h": pred_angles[i, 0].item(),
                            f"angles/sample_{i}_v": pred_angles[i, 1].item(),
                            "iteration": it
                        })

                coordinate_viz_path = visualize_coordinate_system(generator, out_dir, it)
                    
                wandb.log({
                    "sample/rgb": [wandb.Image(rgb, caption=f"RGB at iter {it}")],
                    "sample/depth": [wandb.Image(depth, caption=f"Depth at iter {it}")],
                    "sample/acc": [wandb.Image(acc, caption=f"Acc at iter {it}")],
                    "visualization/coordinate_system": wandb.Image(coordinate_viz_path, caption=f"座標系統 {it}"),
                    "epoch_idx": epoch_idx,
                    "iteration": it
                })

            # (i) Backup if necessary
            if ((it + 1) % 50000) == 0:
                print('Saving backup...')
                checkpoint_io.save('model_%08d.pt' % it, it=it, epoch_idx=epoch_idx, save_to_wandb=True)

            # 儲存檢查點
            if time.time() - t0 > config['training']['save_every']:
                checkpoint_io.save(
                    config['training']['model_file'], 
                    it=it, 
                    epoch_idx=epoch_idx,
                    save_to_wandb=True
                )
                t0 = time.time()
                
                if (restart_every > 0 and t0 - tstart > restart_every):
                    return

if __name__ == '__main__':
    main()