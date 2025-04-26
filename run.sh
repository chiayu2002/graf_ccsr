#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=3
#PBS -l place=shared
#PBS -o output0425_reg10_fc.txt				
#PBS -e error042_reg10_fc.txt				
#PBS -N nerf
cd ~/graf250311										

source ~/.bashrc											
conda activate graftest	

module load cuda-12.4										
#python3 123.py	
#python3 eval.py configs/carla.yaml --pretrained --rotation_elevation
python train.py --config /Data/home/vicky/graf250311/configs/default.yaml