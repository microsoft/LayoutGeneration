#!/bin/bash

# Script for Single-Node multi-GPU training

MODE=$1
DATA_DIR=$2
OUT_DIR=$3
TRAINER=$4
NUM_GPU=$5
EVAL_CKPT_TAG=$6

if [[ $TRAINER = "deepspeed" ]]
then
    # DeepSpeed
    COMMAND="deepspeed --master_port 60005"
else
    # Data Parallel
    TRAINER="basic"
    COMMAND="python"
fi

echo $COMMAND

$COMMAND main.py --${MODE} --dataset publaynet \
--tasks refinement,gen_ts,gen_t,completion,ugen,gen_r \
--eval_tasks gen_r \
--refinement_sort_by_pos_before_sort_by_label \
--gen_ts_add_unk_token \
--gen_ts_shuffle_before_sort_by_label \
--gen_t_add_unk_token \
--gen_t_sort_by_pos_before_sort_by_label \
--completion_sort_by_pos \
--completion_sort_by_pos_before_sort_by_label \
--ugen_sort_by_pos \
--ugen_sort_by_pos_before_sort_by_label \
--gen_r_compact \
--gen_r_add_unk_token \
--gen_r_discrete_before_induce_relations \
--gen_r_shuffle_before_sort_by_label \
--max_num_elements 20 \
--gaussian_noise_mean 0.0 \
--gaussian_noise_std 0.01 \
--train_bernoulli_beta 1.0 \
--test_bernoulli_beta 1.0 \
--data_dir ${DATA_DIR} \
--out_dir ${OUT_DIR} \
--num_layers 8 \
--nhead 8 \
--d_model 512 \
--dropout 0.1 \
--share_embedding \
--epoch 200 \
--batch_size 64 \
--eval_batch_size 1 \
--decode_max_length 150 \
--num_pos_embed 400 \
--lr 1e-04 \
--warmup_num_steps 3000 \
--gradient_accumulation 1 \
--bbox_format ltwh \
--discrete_x_grid 128 \
--discrete_y_grid 128 \
--trainer ${TRAINER} \
--deepscale_config ./scripts/ds_config.json \
--eval_seed 500 \
--enable_task_measure \
--eval_interval 50 \
--eval_ckpt_tag ${EVAL_CKPT_TAG} \
--add_sep_token \
--sort_by_dict \
--add_task_prompt \
--partition_training_data \
--partition_training_data_task_buckets "\-1,-1,-2,0,0,-3" \
--load_vocab