import argparse
import os
from os import path
import numpy as np
import time
import copy
import csv
import random
import torch
from torchvision.utils import save_image
import gc

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib
matplotlib.use('Agg')


import sys
sys.path.append('submodules')        # needed to make imports work in GAN_stability

from submodules.GAN_stability.gan_training.checkpoints_mod import CheckpointIO
from graf.gan_training import Evaluator as Evaluator
from graf.config import get_data, build_models, get_render_poses, load_config
from graf.utils import count_trainable_parameters, to_phi, to_theta, get_nsamples, get_zdist
from graf.transforms import ImgToPatch

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

checkpoint_path = '/Data/home/vicky/graf250916/results/column250923_72/chkpts/model_00039999.pt'

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a GAN with different regularization strategies.'
    )
    parser.add_argument('--config', default='/Data/home/vicky/graf250916/results/column250923_72/config.yaml', type=str, help='Path to config file.')
    parser.add_argument('--fid_kid', action='store_true', help='Evaluate FID and KID.')
    parser.add_argument('--create_sample', help='Generate videos with changing camera pose.')
    parser.add_argument('--rotation_elevation', default= True, action='store_true', help='Generate videos with changing camera pose.')
    parser.add_argument('--shape_appearance', action='store_true', help='Create grid image showing shape/appearance variation.')
    parser.add_argument('--pretrained', action='store_true', help='Load pretrained model.')

    args = parser.parse_args()
    config = load_config(args.config)
    config['data']['fov'] = float(config['data']['fov'])

    # Short hands
    batch_size = config['training']['batch_size']
    out_dir = os.path.join(config['training']['outdir'], config['expname'])
    checkpoint_dir = path.join(out_dir, 'chkpts')
    eval_dir = os.path.join(out_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    fid_kid = int(args.fid_kid)

    config['training']['nworkers'] = 0

    def set_random_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True  # 確定性算法
        torch.backends.cudnn.benchmark = False  # 關閉自動優化

    set_random_seed(0)

    # Logger
    checkpoint_io = CheckpointIO(
        checkpoint_dir=checkpoint_dir
    )

    device = torch.device("cuda:0")

    # Dataset
    train_dataset, hwfr = get_data(config)

    config['data']['hwfr'] = hwfr         # add for building generator
    print(train_dataset, hwfr)
    
    val_dataset = train_dataset                 # evaluate on training dataset for GANs
    if args.fid_kid:
        val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=config['training']['nworkers'],
                shuffle=True, pin_memory=False, sampler=None, drop_last=False   # enable shuffle for fid/kid computation
        )

    # Create models
    generator, _ = build_models(config, disc=False)
    print('Generator params: %d' % count_trainable_parameters(generator))

    # Put models on gpu if needed
    generator = generator.to(device)

    # input transform
    img_to_patch = ImgToPatch(generator.ray_sampler, hwfr[:3])

    # Register modules to checkpoint
    checkpoint_io.register_modules(
        **generator.module_dict  # treat NeRF specially
    )

    # Distributions
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'],
                      device=device)
    
    # Evaluator
    evaluator = Evaluator(fid_kid, generator, zdist, None,
                          batch_size=batch_size, device=device)

    # Train
    tstart = t0 = time.time()
    
    # Load checkpoint
    load_dict = checkpoint_io.load(checkpoint_path)
    it = load_dict.get('it', -1)
    epoch_idx = load_dict.get('epoch_idx', -1)

    def create_labels(num_samples, label_value):
        return torch.full((num_samples, 1), label_value)
    
    if args.create_sample:
        with torch.no_grad():  # 評估模式不需要梯度
            N_samples = 8
            N_poses = 20
            render_radius = config['data']['radius']
            if isinstance(render_radius, str):
                render_radius = float(render_radius.split(',')[1])

            angle_positions = [(i/8, 0.5) for i in range(8)]
            label = create_labels(N_samples, 0)

            z = zdist.sample((N_samples,))
            all_rgb = []
            for i, (u, v) in enumerate(angle_positions):
                print(f"處理角度位置 {i}: ({u}, {v})")
                poses = generator.sample_select_pose(u, v)
                rgb, depth, acc = evaluator.create_samples(z.to(device)[i:i+1], label[i:i+1], poses.unsqueeze(0))
                all_rgb.append(rgb.cpu())  # 立即移到 CPU 以節省 GPU 記憶體

                # 清理中間張量
                del depth, acc
                if i % 2 == 0:  # 每 2 個迭代清理一次
                    torch.cuda.empty_cache()

            rgb = torch.cat(all_rgb, dim=0)
            rgb = ((rgb / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
            rgb = rgb.float() / 255
            n_vis = 8
            filename = 'fake_samples_307.png'
            outpath = os.path.join(eval_dir, filename)
            save_image(rgb, outpath, nrow=n_vis)

            # 清理
            del z, label, all_rgb, rgb
            torch.cuda.empty_cache()
            gc.collect()

    if args.rotation_elevation:
        with torch.no_grad():  # 視頻生成不需要梯度
            N_samples = 1
            N_poses = 150
            render_radius = config['data']['radius']
            if isinstance(render_radius, str):
                render_radius = float(render_radius.split(',')[1])

            # compute render poses
            def get_render_poses_rotation_elevation(N_poses=float('inf')):
                """Compute equidistant render poses varying azimuth and polar angle, respectively."""
                range_theta = to_theta(0.5)
                range_phi = (to_phi(config['data']['umin']+0.5), to_phi(config['data']['umax']+0.5))
                N_phi = min(int(range_phi[1] - range_phi[0]), N_poses)
                render_poses_rotation = get_render_poses(render_radius, angle_range=range_phi, theta=range_theta, N=N_phi)
                return {'rotation': render_poses_rotation}

            z = zdist.sample((N_samples,))
            label = create_labels(N_samples, 0)

            for name, poses in get_render_poses_rotation_elevation(N_poses).items():
                outpath = os.path.join(eval_dir, '{}/'.format(name))
                os.makedirs(outpath, exist_ok=True)
                evaluator.make_video(outpath, z, label, poses, as_gif=False)
                torch.cuda.empty_cache()
                gc.collect()

            # 清理
            del z, label
            torch.cuda.empty_cache()
            gc.collect()
    # Evaluation loop
    if args.fid_kid:
        with torch.no_grad():  # FID/KID 計算不需要梯度
            # Specifically generate samples that can be saved
            n_samples = 1000
            label = create_labels(n_samples, 0)  # 創建 label

            print(f'生成 {n_samples} 個樣本用於 FID/KID 計算...')
            ztest = zdist.sample((n_samples,))
            samples, _, _ = evaluator.create_samples(ztest.to(device), label.to(device))
            samples = (samples / 2 + 0.5).mul_(255).clamp_(0, 255).to(torch.uint8)

            # 保存到 CPU 以節省 GPU 記憶體
            samples_cpu = samples.cpu()
            del samples
            torch.cuda.empty_cache()

            filename = 'samples_fid_kid_{:06d}.npy'.format(n_samples)
            outpath = os.path.join(eval_dir, filename)
            np.save(outpath, samples_cpu.numpy())
            print('Saved {} samples to {}.'.format(n_samples, outpath))

            samples = samples_cpu.to(torch.float) / 255

            n_vis = 8
            filename = 'fake_samples.png'
            outpath = os.path.join(eval_dir, filename)
            save_image(samples[:n_vis**2].clone(), outpath, nrow=n_vis)
            print('Plot examples under {}.'.format(outpath))

            filename = 'real_samples.png'
            outpath = os.path.join(eval_dir, filename)
            sample, _ = get_nsamples(val_loader, n_vis**2)
            real = sample / 2 + 0.5
            save_image(real[:n_vis ** 2].clone(), outpath, nrow=n_vis)
            print('Plot examples under {}.'.format(outpath))

            del real, sample
            torch.cuda.empty_cache()

            # Compute FID and KID
            print('計算 FID 和 KID...')
            fid_cache_file = os.path.join(out_dir, 'fid_cache_train.npz')
            kid_cache_file = os.path.join(out_dir, 'kid_cache_train.npz')
            evaluator.inception_eval.initialize_target(val_loader, cache_file=fid_cache_file, act_cache_file=kid_cache_file)

            samples = samples * 2 - 1
            sample_loader = torch.utils.data.DataLoader(
                samples,
                batch_size=evaluator.batch_size,
                num_workers=min(config['training']['nworkers'], 4),
                shuffle=False, pin_memory=False, sampler=None, drop_last=False
            )
            fid, kid = evaluator.compute_fid_kid(label, sample_loader)

            filename = 'fid_kid.csv'
            outpath = os.path.join(eval_dir, filename)
            with open(outpath, mode='w') as csv_file:
                fieldnames = ['fid', 'kid']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({'fid': fid, 'kid': kid})

            print('Saved FID ({:.1f}) and KIDx100 ({:.2f}) to {}.'.format(fid, kid*100, outpath))

            # 最終清理
            del ztest, label, samples, samples_cpu, sample_loader
            torch.cuda.empty_cache()
            gc.collect()