#!/usr/bin/env bash 

### To run experiment, modify
GPU_ID=0
task_name=Biden_base_Trump_target # choose from: Biden_base_Trump_target, healthyFood_base_hamburgerFries_target, kidSports_base_kidVideoGame_target, lowFuelLight_base_engineLight_target


CUDA_VISIBLE_DEVICES=$GPU_ID python poison_llava.py \
 --task_data_pth data/task_data/$task_name --poison_save_pth data/poisons/llava/$task_name \
 --iter_attack 4000 --lr_attack 0.2 --diff_aug_specify None \
 --batch_size 60 