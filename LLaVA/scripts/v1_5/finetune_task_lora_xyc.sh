#!/bin/bash

# bash LLaVA/scripts/v1_5/finetune_task_lora_xyc.sh 

poison=cc_sbu_align-healthyFood_base_hamburgerFries_target-poison_200-seed_0-llava # change this and GPU id (and accumulation) and master_port
GPU_ID=3

master_port=$(( $GPU_ID+ 29000 ))
# if 2 GPU, acc_step=4; if 1 GPU, acc_step=8 
# effective bs = per_device_bs * accumulation * num_GPU should be 128 
gradient_accumulation_steps=8

### no need to change the following
num_train_epochs=1 
per_device_train_batch_size=16

data_root=/cmlscratch/xic/Poisoning-Vision-Language-Models/data/
data_path=$data_root/poisoned_training_data/$poison.json
output_dir=./checkpoints/llava-v1.5-7b-task-lora__$poison-epoch_$num_train_epochs

deepspeed --include localhost:$GPU_ID --master_port $master_port LLaVA/llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed LLaVA/scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path $data_path \
    --image_folder $data_root \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb