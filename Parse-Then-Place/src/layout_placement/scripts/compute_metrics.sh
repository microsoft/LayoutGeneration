#!/bin/bash

data_dir=$1
result_write_to=$2
dataset_name=$3

if [[ ! -d ${result_write_to} ]]
then
    mkdir -p ${result_write_to}
fi

PYTHONPATH=$(pwd) \
python layout_placement/eval.py \
--train_source_file train.json \
--test_ground_truth_file test.json \
--data_dir ${data_dir} \
--out_dir ${result_write_to} \
--dataset_name ${dataset_name}
