#!/bin/bash

ckpt_path=$1
data_dir=$2
result_write_to=$3
dataset_name=$4

if [[ ! -d ${result_write_to} ]]
then
    mkdir -p ${result_write_to}
fi

PYTHONPATH=$(pwd) \
python layout_placement/test.py \
--model_name_or_path ${ckpt_path} \
--tokenizer_path ${ckpt_path} \
--train_source_file train.json \
--test_ground_truth_file test.json \
--stage_one_prediction_file test.json \
--data_dir ${data_dir} \
--out_dir ${result_write_to} \
--batch_size 16 \
--dataset_name ${dataset_name} \
--is_two_stage false \