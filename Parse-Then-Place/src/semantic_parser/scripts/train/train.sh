#!/bin/bash

data_dir=$1
output_dir=$2
eval_split=$3
model_name=$4
description=$5
dataset=$6          # rico or web
training_from_scratch=$7

now=$(date +"%Y%m%d%H%M")
output_dir=$output_dir/$description-$now
mkdir -p $output_dir

# use wandb
export WANDB_NAME="nl2layout-sp-$description-$now-$dataset"

BASE_PATH='./semantic_parser'

if [ "${eval_split}" != "val" ] && [ "${eval_split}" != "train" ]
then
    eval_split="test"
fi

if [ -z "$model_name" ]
then
    model_name="google/t5-v1_1-small"
fi

if [ "${training_from_scratch}" = "true" ]
then
    additional_param="--training_from_scratch"
else
    additional_param=""
fi

echo "Create experiment: ${description}"
echo "Using model: ${model_name}"

PYTHONPATH=$(pwd) \
deepspeed ${BASE_PATH}/run_parser.py \
--seed 100 \
--do_train \
--data_dir ${data_dir} \
--ir_remove_value \
--replace_explicit_value \
--output_dir ${output_dir} \
--model_name ${model_name} \
--generation_max_length 600 \
--num_epochs 200 \
--eval_micro_batch_size 16 \
--eval_delay 5 \
--save_interval 5 \
--deepscale_config ${BASE_PATH}/scripts/train/ds_config.json \
--log_level info \
--eval_split ${eval_split} \
--label_smoothing_factor 0.1 \
--dataset_name ${dataset} \
${additional_param} \
