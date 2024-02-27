#!/bin/bash
set -e

GPU_ID=0
task_name=Biden_base_Trump_target # choose from: Biden_base_Trump_target, healthyFood_base_hamburgerFries_target, kidSports_base_kidVideoGame_target, lowFuelLight_base_engineLight_target
model_setting=llava # llava, or instructBLIP_to_llava, miniGPT4v2_to_llava, llava_jpeg, llava_aug_lavisCLIP, llava_augTrain_lavisCLIP

declare -a num_poison_list=(200 0 5 10 20 30 50 100 150) 
if [[ "$task_name" == "lowFuelLight_base_engineLight_target" ]]
then
      declare -a num_poison_list=(178 0 5 10 20 30 50 100 150) 
fi

### the following is automatic
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )" # the directory of this script
clean_data_name=cc_sbu_align
model_base=liuhaotian/llava-v1.5-7b # for lora

gpu_list=$GPU_ID 
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

SPLIT="llava_gqa_testdev_balanced"
GQADIR="LLaVA/playground/data/eval/gqa/data"  # the directory of the GQA dataset

for num_poison in "${num_poison_list[@]}"
do
    model_path=checkpoints/$model_setting/$clean_data_name-$task_name/poison_$num_poison-seed_0
    echo Benchmarking GQA: $model_path

    # generate predictions
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m LLaVA.llava.eval.model_vqa_loader \
            --model-path $model_path --model-base $model_base \
            --question-file LLaVA/playground/data/eval/gqa/$SPLIT.jsonl \
            --image-folder LLaVA/playground/data/eval/gqa/data/images \
            --answers-file $model_path/eval/gqa/answers/$SPLIT/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0 \
            --conv-mode vicuna_v1 &
    done
    wait

    output_file=$model_path/eval/gqa/answers/$SPLIT/merge.jsonl
    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat $model_path/eval/gqa/answers/$SPLIT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

    python LLaVA/scripts/convert_gqa_for_eval.py --src $output_file --dst $model_path/eval/gqa/answers/$SPLIT/testdev_balanced_predictions.json

    echo '********************************************'
    result_save_pth=$SCRIPTPATH/$model_path/eval/gqa/result.log 
    cd $GQADIR
    # python eval/eval.py --tier testdev_balanced | tee $SCRIPTPATH/$model_path/eval/gqa/result.log 
    python eval/eval.py --tier testdev_balanced --predictions $SCRIPTPATH/$model_path/eval/gqa/answers/$SPLIT/testdev_balanced_predictions.json | grep Accuracy: | tee -a $result_save_pth
    echo $model_path | tee -a $result_save_pth

    cd $SCRIPTPATH
done