#!/bin/bash

mode=$1
data_dir=$2
output_dir=$3
eval_split=$4
model_name=$5

DATA_VERSION='v5.1'
BASE_PATH='./semantic_parser'

if [ "${eval_split}" != "val" ] && [ "${eval_split}" != "train" ]
then
    eval_split="test"
fi

if [ -z "$model_name" ]
then
    model_name="google/t5-v1_1-small"
fi
echo "Using model: ${model_name}"

deepspeed ${BASE_PATH}/run_parser.py \
--seed 100 \
--do_${mode} \
--data_dir ${data_dir}/${DATA_VERSION} \
--output_dir ${output_dir} \
--model_name ${model_name} \
--generation_max_length 400 \
--num_epochs 1500 \
--eval_micro_batch_size 16 \
--eval_delay 1 \
--deepscale_config ${BASE_PATH}/scripts/prompt_tuning_ds_config.json \
--log_level info \
--eval_split ${eval_split} \
--use_adafactor \
--adafactor_lr 0.3 \
--tuning_method prompt_tuning \
--num_prompt_tokens 100 \
--prompt_init_method vocab
