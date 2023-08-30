#!/bin/bash

data_dir=$1
output_dir=$2
num_gpus=$3
add_complete_token=$4
run_description=$5

if [[ -z ${add_complete_token} ]]
then
    add_complete_token="true"
fi

if [[ -z ${run_description} ]]
then
    run_description=$(date +%Y/%m/%d-%H:%M:%S)
fi

if [[ ! -d ${output_dir} ]]
then
    mkdir -p ${output_dir}
fi

PYTHONPATH=$(pwd) \
python -m torch.distributed.launch --nproc_per_node=${num_gpus} layout_placement/train.py \
--batch_size 4 \
--lr 1e-4 \
--epochs 250 \
--data_dir ${data_dir} \
--out_dir ${output_dir} \
--model_name_or_path t5-base \
--tokenizer_path t5-base \
--train_source_file train.json \
--val_source_file val.json \
--evaluation_interval 1 \
--scheduler warmupconstant \
--run_description ${run_description} \
--add_complete_token ${add_complete_token} \
--is_two_stage false