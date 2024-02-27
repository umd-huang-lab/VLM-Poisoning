#!/bin/bash

### vizwiz
# after run the following script, we need to submit to the evaluator website to get final metrics.
# Step0: run the generation
# Step1: (using evalai env) evalai challenge 1911 phase 3812 submit --large --private --file $model_path/eval/vizwiz/answers/answers_upload.json
# Step2: on website, choose test-dev2023-VQA and download the csv file. Paste the link in the result file column and result will be there.
# website:https://eval.ai/web/challenges/challenge-page/1911/my-submission

GPU_ID=0
task_name=Biden_base_Trump_target # choose from: Biden_base_Trump_target, healthyFood_base_hamburgerFries_target, kidSports_base_kidVideoGame_target, lowFuelLight_base_engineLight_target
model_setting=llava # llava, or instructBLIP_to_llava, miniGPT4v2_to_llava, llava_jpeg, llava_augTrain_lavisCLIP

declare -a num_poison_list=(200 0 5 10 20 30 50 100 150) 
if [[ "$task_name" == "lowFuelLight_base_engineLight_target" ]]
then
      declare -a num_poison_list=(178 0 5 10 20 30 50 100 150) 
fi


### the following is automatic
clean_data_name=cc_sbu_align
model_base=liuhaotian/llava-v1.5-7b # for lora

for num_poison in "${num_poison_list[@]}"
do
    model_path=checkpoints/$model_setting/$clean_data_name-$task_name/poison_$num_poison-seed_0
    echo Benchmarking vizwiz: $model_path

    ### generate prediction (comment "upload to evalai")
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m LLaVA.llava.eval.model_vqa_loader \
        --model-path $model_path --model-base $model_base \
        --question-file LLaVA/playground/data/eval/vizwiz/llava_test.jsonl \
        --image-folder LLaVA/playground/data/eval/vizwiz/test \
        --answers-file $model_path/eval/vizwiz/answers/answers_file.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1

    python LLaVA/scripts/convert_vizwiz_for_submission.py \
        --annotation-file LLaVA/playground/data/eval/vizwiz/llava_test.jsonl \
        --result-file $model_path/eval/vizwiz/answers/answers_file.jsonl \
        --result-upload-file $model_path/eval/vizwiz/answers/answers_upload.json

    ### upload to evalai (comment "generate prediction"); you may need to use another environment evalai env from evalai
    # echo Submitting to evalai results of $model_path
    # evalai challenge 1911 phase 3812 submit --large --private --file $model_path/eval/vizwiz/answers/answers_upload.json 
    
done