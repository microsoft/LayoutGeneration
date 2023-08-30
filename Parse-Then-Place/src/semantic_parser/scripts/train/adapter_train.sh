#!/bin/bash

data_dir=$1
output_dir=$2
eval_split=$3
model_name=$4
description=$5
dataset=$6          # rico or web

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
--num_epochs 150 \
--eval_micro_batch_size 16 \
--eval_delay 5 \
--save_interval 5 \
--deepscale_config ${BASE_PATH}/scripts/train/adapter_ds_config.json \
--log_level info \
--eval_split ${eval_split} \
--label_smoothing_factor 0.1 \
--tuning_method adapter \
--adapter_name parsing \
--adapter_type Parallel \
--adapter_non_linearity relu \
--adapter_reduction_factor 16 \
--adapter_scaling 4 \
--learning_rate 3e-3 \
--weight_decay 0.0 \
--dataset_name ${dataset}


# convert checkpoint
# if [ "${mode}" == "train" ]
# then
#     python ${output_dir}/checkpoint/zero_to_fp32.py ${output_dir}/checkpoint ${output_dir}/checkpoint/pytorch_model.bin
# fi