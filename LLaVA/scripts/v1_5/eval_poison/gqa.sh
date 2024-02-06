#!/bin/bash

set -e

### usage: CUDA_VISIBLE_DEVICES=3 bash LLaVA/scripts/v1_5/eval_poison/gqa.sh

CKPT=llava-v1.5-7b-task-lora__cc_sbu_align-Biden_base_Trump_target-poison_50-seed_0-llava-epoch_1
# CKPT=llava-v1.5-7b # using liuhaotian/llava-v1.5-7b; need to disable --model-base $model_base in the following

if [ $CKPT == 'llava-v1.5-7b' ]
then
  model_path='liuhaotian/llava-v1.5-7b'
else
  model_path=checkpoints/$CKPT
fi
echo $model_path

model_base=liuhaotian/llava-v1.5-7b # lora support
LLaVA_root=LLaVA # relative directory to LLaVA/
###
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="llava_gqa_testdev_balanced"
GQADIR="$LLaVA_root/playground/data/eval/gqa/data" 

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m $LLaVA_root.llava.eval.model_vqa_loader \
        --model-path $model_path --model-base $model_base \
        --question-file $LLaVA_root/playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder $LLaVA_root/playground/data/eval/gqa/data/images \
        --answers-file $LLaVA_root/playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$LLaVA_root/playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $LLaVA_root/playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python $LLaVA_root/scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json # will re-write this every time

echo '********************************************'
echo $model_path

cd $GQADIR
python eval/eval.py --tier testdev_balanced

echo $model_path
