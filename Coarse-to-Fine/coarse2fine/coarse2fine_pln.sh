#!/bin/bash

# Script for Single-Node multi-GPU training

MODE=$1
DATA_DIR=$2
OUT_DIR=$3
TRAINER=$4
NUM_GPU=$5

TASK_DIR='./coarse2fine'

if [[ $TRAINER = "deepspeed" ]]
then
    # DeepSpeed
    COMMAND="deepspeed --master_port 60001"
elif [[ $TRAINER = "ddp" ]]
then
    # Distributed Data Parallel
    COMMAND="python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port 6870"
else
    # Data Parallel
    TRAINER="basic"
    COMMAND="python"
fi

echo $COMMAND

$COMMAND ${TASK_DIR}/main.py --${MODE} --dataset publaynet \
--max_num_elements 20 \
--num_labels 5 \
--data_dir ${DATA_DIR} \
--out_dir ${OUT_DIR} \
--epoch 100 \
--batch_size 128 \
--eval_batch_size 128 \
--lr 0.0008 \
--kl_start_step 1000 \
--kl_end_step 60000 \
--gradient_accumulation 1 \
--bbox_format ltwh \
--discrete_x_grid 128 \
--discrete_y_grid 128
