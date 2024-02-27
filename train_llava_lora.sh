#!/bin/bash

##### argument to modify
# for llava_augTrain_lavisCLIP (training llava using data augmentation), need to modify LLaVA/llava/model/multimodal_encoder/clip_encoder.py; don't run other llava related experiments in the mean time.
GPU_ID=0
task_name=Biden_base_Trump_target # choose from: Biden_base_Trump_target, healthyFood_base_hamburgerFries_target, kidSports_base_kidVideoGame_target, lowFuelLight_base_engineLight_target
model_setting=llava # choose from: llava, instructBLIP_to_llava, miniGPT4v2_to_llava, llava_jpeg, llava_aug_lavisCLIP, llava_augTrain_lavisCLIP, llava_aug_lavisCLIP_jpeg, llava_augTrainLavisCLIP_noAugPoison, llava_aug_jpeg_jpeg
seed=0

SAVE_ROOT=. # change to your root for saving the poisoned VLMs

# modify num_poison_list to train on different number of poisoned samples
declare -a num_poison_list=(200 0 5 10 20 30 50 100 150 200) 
if [[ "$task_name" == "lowFuelLight_base_engineLight_target" ]]
then
      declare -a num_poison_list=(178 0 5 10 20 30 50 100 150 178) 
fi


##### the following are automatic
clean_data_name=cc_sbu_align
master_port=$(( $GPU_ID+ 29000 ))
# if 2 GPU, acc_step=4; if 1 GPU, acc_step=8; effective bs = per_device_bs * accumulation * num_GPU should be 128 
gradient_accumulation_steps=8
per_device_train_batch_size=16
num_train_epochs=1

data_root=./data 
data_path_root=$data_root/poisoned_training_data/$model_setting/$clean_data_name-$task_name
output_dir_root=$SAVE_ROOT/checkpoints/$model_setting/$clean_data_name-$task_name

for num_poison in "${num_poison_list[@]}"
do
    
    data_path=$data_path_root/poison_$num_poison-seed_$seed.json  
    output_dir=$output_dir_root/poison_$num_poison-seed_$seed/

    echo GPU=$GPU_ID training: $output_dir
 
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
done

