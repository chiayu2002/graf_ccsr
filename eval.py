import argparse
import os
from os import path
import numpy as np
import time
import copy
import csv
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torchvision.utils import save_image

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

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# import wandb
# run = wandb.init()
# artifact = run.use_artifact('vicky20020808/graftest/model_checkpoint:v81', type='checkpoint')
# artifact_dir = artifact.download()
checkpoint_path = '/Data/home/vicky/graf250218/results/4column250222_rename/chkpts/model.pt'

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a GAN with different regularization strategies.'
    )
    parser.add_argument('--config', default='/Data/home/vicky/graf250218/results/4column250222_rename/config.yaml', type=str, help='Path to config file.')
    parser.add_argument('--fid_kid', action='store_true', help='Evaluate FID and KID.')
    parser.add_argument('--create_sample', default= True, action='store_true', help='Generate videos with changing camera pose.')
    parser.add_argument('--rotation_elevation', action='store_true', help='Generate videos with changing camera pose.')
    parser.add_argument('--shape_appearance', action='store_true', help='Create grid image showing shape/appearance variation.')
    parser.add_argument('--pretrained', action='store_true', help='Load pretrained model.')
    # parser.add_argument('--reconstruction', action='store_true', help='Generate images and run COLMAP for 3D reconstruction.')

    args = parser.parse_args()
    config = load_config(args.config)
    config['data']['fov'] = float(config['data']['fov'])

    # Short hands
    batch_size = config['training']['batch_size']
    out_dir = os.path.join(config['training']['outdir'], config['expname'])
    if args.pretrained:
        config['expname'] = '%s_%s' % (config['data']['type'], config['data']['imsize'])
        out_dir = os.path.join(config['training']['outdir'], config['expname'] + '_from_pretrained')
    checkpoint_dir = path.join(out_dir, 'chkpts')
    eval_dir = os.path.join(out_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    fid_kid = int(args.fid_kid)

    config['training']['nworkers'] = 0

    # Logger
    checkpoint_io = CheckpointIO(
        checkpoint_dir=checkpoint_dir
    )

    device = torch.device("cuda:0")

    # Dataset
    train_dataset, hwfr = get_data(config)
    # in case of orthographic projection replace focal length by far-near
    if config['data']['orthographic']:
        hw_ortho = (config['data']['far']-config['data']['near'], config['data']['far']-config['data']['near'])
        hwfr[2] = hw_ortho

    config['data']['hwfr'] = hwfr         # add for building generator
    print(train_dataset, hwfr)
    
    val_dataset = train_dataset                 # evaluate on training dataset for GANs
    if args.fid_kid:
        val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=config['training']['nworkers'],
                shuffle=True, pin_memory=False, sampler=None, drop_last=False,   # enable shuffle for fid/kid computation
                generator=torch.Generator(device='cuda:0')
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


    # Get model file
    # if args.pretrained:
    #     config_pretrained = load_config('configs/pretrained_models.yaml', 'configs/pretrained_models.yaml')
    #     model_file = config_pretrained[config['data']['type']][config['data']['imsize']]
    # else:
    #     model_file = 'model_checkpoint.pth'

    # Distributions
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'],
                      device=device)
    
    # Test generator, use model average
    generator_test = copy.deepcopy(generator)
    generator_test.parameters = lambda: generator_test._parameters
    generator_test.named_parameters = lambda: generator_test._named_parameters
    checkpoint_io.register_modules(**{k + '_test': v for k, v in generator_test.module_dict.items()})

    # Evaluator
    evaluator = Evaluator(fid_kid, generator_test, zdist, None,
                          batch_size=batch_size, device=device)

    # Train
    tstart = t0 = time.time()
    
    # Load checkpoint
    load_dict = checkpoint_io.load(checkpoint_path)
    it = load_dict.get('it', -1)
    epoch_idx = load_dict.get('epoch_idx', -1)

    def create_labels(num_samples, label_value):
        return torch.full((num_samples, 1), label_value)
    
    N_samples = 1
    label = create_labels(8, 0)
    
    # Evaluation loop
    if args.fid_kid:
        # Specifically generate samples that can be saved
        n_samples = 1000
        ztest = zdist.sample((n_samples,))
        samples, _, _ = evaluator.create_samples(ztest.to(device), label.to(device))
        samples = (samples / 2 + 0.5).mul_(255).clamp_(0, 255).to(torch.uint8)      # convert to unit8

        filename = 'samples_fid_kid_{:06d}.npy'.format(n_samples)
        outpath = os.path.join(eval_dir, filename)
        np.save(outpath, samples.numpy())
        print('Saved {} samples to {}.'.format(n_samples, outpath))

        samples = samples.to(torch.float) / 255

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

        # Compute FID and KID
        fid_cache_file = os.path.join(out_dir, 'fid_cache_train.npz')
        kid_cache_file = os.path.join(out_dir, 'kid_cache_train.npz')
        evaluator.inception_eval.initialize_target(val_loader, cache_file=fid_cache_file, act_cache_file=kid_cache_file)

        samples = samples * 2 - 1
        sample_loader = torch.utils.data.DataLoader(
            samples,
            batch_size=evaluator.batch_size, num_workers=config['training']['nworkers'],
            shuffle=False, pin_memory=False, sampler=None, drop_last=False,
            generator=torch.Generator(device='cuda:0')
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

    if args.rotation_elevation:
        N_samples = 8
        N_poses = 20            # corresponds to number of frames
        render_radius = config['data']['radius']
        if isinstance(render_radius, str):  # use maximum radius
            render_radius = float(render_radius.split(',')[1])

        # compute render poses
        def get_render_poses_rotation_elevation(N_poses=float('inf')):
            """Compute equidistant render poses varying azimuth and polar angle, respectively."""
            range_theta = (to_theta(config['data']['vmin']), to_theta(config['data']['vmax']))
            range_phi = (to_phi(config['data']['umin']), to_phi(config['data']['umax']))

            theta_mean = 0.5 * sum(range_theta)
            phi_mean = 0.5 * sum(range_phi)

            N_theta = min(int(range_theta[1] - range_theta[0]), N_poses)  # at least 1 frame per degree
            N_phi = min(int(range_phi[1] - range_phi[0]), N_poses)  # at least 1 frame per degree

            render_poses_rotation = get_render_poses(render_radius, angle_range=range_phi, theta=theta_mean, N=N_phi)
            render_poses_elevation = get_render_poses(render_radius, angle_range=range_theta, theta=phi_mean, N=N_theta,
                                                      swap_angles=True)

            return {'rotation': render_poses_rotation, 'elevation': render_poses_elevation}

        z = zdist.sample((N_samples,))

        for name, poses in get_render_poses_rotation_elevation(N_poses).items():
            outpath = os.path.join(eval_dir, '{}/'.format(name))
            os.makedirs(outpath, exist_ok=True)
            evaluator.make_video(outpath, z, label, poses, as_gif=False)
            torch.cuda.empty_cache()

    if args.create_sample:
        N_samples = 8
        N_poses = 20            # corresponds to number of frames
        render_radius = config['data']['radius']
        if isinstance(render_radius, str):  # use maximum radius
            render_radius = float(render_radius.split(',')[1])

        # compute render poses
        # def get_render_poses_by_angles(render_radius, azimuth, elevation, N_poses=1):
        #     """Compute equidistant render poses varying azimuth and polar angle, respectively."""
        #     # theta = to_theta(u)
        #     # phi = to_phi(v)
        #     angle_range = (azimuth, azimuth)

        #     render_poses_rotation = get_render_poses(render_radius, angle_range=angle_range, theta=elevation, N=N_poses)

        #     return  render_poses_rotation
        
        angle_positions = [
                    (0., 0.5),    
                    (0.125, 0.5),   
                    (0.25, 0.5),   
                    (0.375, 0.5), 
                    (0.5, 0.5),  
                    (0.625, 0.5),  
                    (0.75, 0.5),  
                    (0.875, 0.5)  
                ] 

        z = zdist.sample((N_samples,))
        all_rgb = []
        for i, (u, v) in enumerate(angle_positions):
            # position_angle = (azimuth + 180) % 360
            # print(f"指定角度:{u}, 轉換後角度:{position_angle}")
            poses = generator.sample_select_pose(u ,v)
            # print(poses)
            rgb, depth, acc = evaluator.create_samples(z.to(device)[i:i+1], label[i:i+1], poses.unsqueeze(0))
            all_rgb.append(rgb)

        rgb = torch.cat(all_rgb, dim=0)
        rgb = ((rgb / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
        rgb = rgb.float() / 255        
        n_vis = 8
        filename = 'fake_samples_123.png'
        outpath = os.path.join(eval_dir, filename)
        save_image(rgb, outpath, nrow=n_vis)

    if args.shape_appearance:
        N_shapes = 5
        N_appearances = 5

        # constant pose
        pose = render_poses[len(render_poses) // 2]
        pose = pose.unsqueeze(0).expand(N_shapes * N_appearances, -1, -1)

        # sample shape latent codes
        z_shape = zdist.sample((N_shapes, 1))[..., :config['z_dist']['dim'] - config['z_dist']['dim_appearance']]
        z_shape = z_shape.expand(-1, N_appearances, -1)

        z_appearance = zdist.sample((1, N_appearances,))[..., config['z_dist']['dim_appearance']:]
        z_appearance = z_appearance.expand(N_shapes, -1, -1)

        z_grid = torch.cat([z_shape, z_appearance], dim=-1).flatten(0, 1)

        rgbs, _, _ = evaluator.create_samples(z_grid, poses=pose)
        rgbs = rgbs / 2 + 0.5

        outpath = os.path.join(eval_dir, 'shape_appearance.png')
        save_image(rgbs, outpath, nrow=N_shapes, padding=0)

    # if args.reconstruction:

    #     N_samples = 8
    #     N_poses = 400            # corresponds to number of frames
    #     ztest = zdist.sample((N_samples,))

    #     # sample from mean radius
    #     radius_orig = generator_test.radius
    #     if isinstance(radius_orig, tuple):
    #         generator_test.radius = 0.5 * (radius_orig[0]+radius_orig[1])

    #     # output directories
    #     rec_dir = os.path.join(eval_dir, 'reconstruction')
    #     image_dir = os.path.join(rec_dir, 'images')
    #     colmap_dir = os.path.join(rec_dir, 'models')

    #     # generate samples and run reconstruction
    #     for i, z_i in enumerate(ztest):
    #         outpath = os.path.join(image_dir, 'object_{:04d}'.format(i))
    #         os.makedirs(outpath, exist_ok=True)

    #         # create samples
    #         z_i = z_i.reshape(1,-1).repeat(N_poses, 1)
    #         rgbs, _, _ = evaluator.create_samples(z_i.to(device))
    #         rgbs = rgbs / 2 + 0.5
    #         for j, rgb in enumerate(rgbs):
    #             save_image(rgb.clone(), os.path.join(outpath, '{:04d}.png'.format(j)))

    #         # run COLMAP for 3D reconstruction
    #         colmap_input_dir = os.path.join(image_dir, 'object_{:04d}'.format(i))
    #         colmap_output_dir = os.path.join(colmap_dir, 'object_{:04d}'.format(i))
    #         colmap_cmd = './external/colmap/run_colmap_automatic.sh {} {}'.format(colmap_input_dir, colmap_output_dir)
    #         print(colmap_cmd)
    #         os.system(colmap_cmd)

    #         # filter out white points
    #         filter_ply(colmap_output_dir)

        # reset radius for generator
        # generator_test.radius = radius_orig
