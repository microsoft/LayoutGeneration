#!/bin/bash

data_dir=$1
ckpt_dir=$2
prediction_dir=$3
eval_split=$4
model_name=$5
dataset_name=$6
ckpt_tag=$7

BASE_PATH='./semantic_parser'

if [[ ! -d ${prediction_dir} ]]
then
    mkdir -p ${prediction_dir}
fi

if [ ! -z "$ckpt_tag" ]
then
    python ${BASE_PATH}/zero_to_fp32.py ${ckpt_dir} ${ckpt_dir}/pytorch_model.bin ${ckpt_tag}
fi

if [ "${eval_split}" != "val" ] && [ "${eval_split}" != "train" ]
then
    eval_split="test"
fi

if [ -z "$model_name" ]
then
    model_name="google/t5-v1_1-small"
fi
echo "Using model: ${model_name}"

PYTHONPATH=$(pwd) \
deepspeed ${BASE_PATH}/run_parser.py \
--seed 100 \
--do_eval \
--data_dir ${data_dir} \
--ir_remove_value \
--replace_explicit_value \
--output_dir ${ckpt_dir} \
--prediction_dir ${prediction_dir} \
--model_name ${model_name} \
--generation_max_length 600 \
--num_epochs 100 \
--eval_micro_batch_size 16 \
--eval_delay 3 \
--save_interval 3 \
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
--dataset_name ${dataset_name}
