#!/bin/bash

data_dir=$1
output_dir=$2
num_gpus=$3
training_set_size=$4
run_description=$5

if [[ -z ${training_set_size} ]]
then
    training_set_size=1500000
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
--epochs 200 \
--data_dir ${data_dir} \
--out_dir ${output_dir} \
--num_train ${training_set_size} \
--model_name_or_path t5-base \
--tokenizer_path t5-base \
--train_source_file train_source.txt \
--train_target_file train_target.txt \
--val_source_file val_source.txt \
--val_target_file val_target.txt \
--scheduler warmuplinear \
--run_description ${run_description}
