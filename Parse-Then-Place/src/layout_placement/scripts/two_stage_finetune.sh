#!/bin/bash

data_dir=$1
output_dir=$2
pretrain_ckpt=$3
num_gpus=$4
add_complete_token=$5
run_description=$6

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
--lr 5e-5 \
--epochs 250 \
--data_dir ${data_dir} \
--out_dir ${output_dir} \
--model_name_or_path ${pretrain_ckpt} \
--tokenizer_path ${pretrain_ckpt} \
--train_source_file train.json \
--val_source_file val.json \
--evaluation_interval 5 \
--scheduler warmupconstant \
--run_description ${run_description} \
--add_complete_token ${add_complete_token} \
# --num_train 1000