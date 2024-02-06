#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1,2,3 bash LLaVA/scripts/v1_5/eval_poison/vizwiz.sh

# (eval using evalai env) evalai challenge 1911 phase 3812 submit --large --private --file answers_upload/xxx.json 

model_path=llava-v1.5-7b-task-lora__cc_sbu_align-Biden_base_Trump_target-poison_100-seed_0-llava


LLaVA_root=LLaVA # relative directory to LLaVA/
output_answers_file=$model_path.jsonl
model_base=liuhaotian/llava-v1.5-7b

python -m $LLaVA_root.llava.eval.model_vqa_loader \
    --model-path checkpoints/$model_path --model-base $model_base \
    --question-file $LLaVA_root/playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder $LLaVA_root/playground/data/eval/vizwiz/test \
    --answers-file $LLaVA_root/playground/data/eval/vizwiz/answers/$output_answers_file \
    --temperature 0 \
    --conv-mode vicuna_v1

python $LLaVA_root/scripts/convert_vizwiz_for_submission.py \
    --annotation-file $LLaVA_root/playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file $LLaVA_root/playground/data/eval/vizwiz/answers/$output_answers_file \
    --result-upload-file $LLaVA_root/playground/data/eval/vizwiz/answers_upload/${output_answers_file::-1}
