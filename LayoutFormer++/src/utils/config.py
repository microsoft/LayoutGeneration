import os
import deepspeed


def add_arguments(parser):

    parser.add_argument('--test', dest='test', action='store_true',
                        help='whether to do inference')
    parser.add_argument('--train', dest='test', action='store_false',
                        help='whether to do training')

    # data dir
    parser.add_argument('--data_dir', type=str, default='../datasets/',
                        help='data dir')
    parser.add_argument('--out_dir', type=str, default='../output/',
                        help='output directory')

    # data
    parser.add_argument('--dataset', type=str, choices=['rico', 'publaynet'],
                        help='dataset name')
    parser.add_argument('--max_num_elements', type=int, default=20,
                        help='max number of design elements')

    # hyperparameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size for each GPU')
    parser.add_argument('--train_log_step', type=int, default=50,
                        help='number of steps to log loss during training')
    parser.add_argument('--seed', type=int, help='manual seed')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate')
    parser.add_argument('--warmup_num_steps', type=int, default=0, help='learning warmup steps')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout rate')
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                        help='gradient accumulation step')
    parser.add_argument('--enable_clip_gradient', action='store_true', default=False,
                        help='whether to enable gradient clip')
    parser.add_argument('--clip_gradient', type=float, default=5,
                        help='clips gradient norm of an iterable of parameters.')
    parser.add_argument('--epoch', type=int, default=30, help='number of epoch to train')
    parser.add_argument('--adafactor', action='store_true', default=False,
                        help='whether to use adafactor optimizer')

    # Data Format
    parser.add_argument('--shuffle_elements', action='store_true', default=False,
                        help='whehter to randomly shuffle elements')  # shuffle once is ok
    parser.add_argument('--bbox_format', type=str, choices=['xywh', 'ltrb', 'ltwh'], default='ltwh',
                        help='bbox coordinate ')
    parser.add_argument('--discrete_x_grid', type=int, default=64,
                        help='number of grids in x axis')
    parser.add_argument('--discrete_y_grid', type=int, default=64,
                        help='number of grids in y axis')

    # Inference
    parser.add_argument('--num_save', type=int, default=100,
                        help='number of layouts to save as images')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='batch sized used in inference')

    # FID Model
    parser.add_argument('--fid_max_num_element', type=int, help='max number of elements for FID model')
    parser.add_argument('--fid_num_labels', type=int, help='number of element labels for FID model')


def add_task_arguments(parser):
    add_multitask_arguments(parser)

    # Model
    parser.add_argument('--num_layers', type=int, default=8,
                        help='num_layers for generator')
    parser.add_argument('--nhead', type=int, default=8,
                        help='nhead for generator')
    parser.add_argument('--d_model', type=int, default=512,
                        help='d_model for generator')
    parser.add_argument('--share_embedding', action='store_true',
                        help='whether to share embeddings between encoder and decoder')
    parser.add_argument('--num_pos_embed', type=int, default=150,
                        help='number of position encoding')
    parser.add_argument('--add_task_embedding', action='store_true', default=False,
                        help='add task embedding to the output of transformer encoder')
    parser.add_argument('--add_task_prompt_token_in_model', action='store_true', default=False,
                        help='add task prompt tokens in transformer')
    parser.add_argument('--num_task_prompt_token', type=int, default=1,
                        help='number of task prompt tokens added in transformer')

    # Inference
    parser.add_argument('--load_vocab', default=False, action='store_true',
                        help='whether to load vocab')
    parser.add_argument('--eval_tasks', required=False, type=str,
                        help='inference tasks in multitask')


def add_multitask_arguments(parser):

    # Task format
    parser.add_argument('--add_sep_token', action='store_true', help='Add sep token between elements')
    parser.add_argument('--add_task_prompt', action='store_true', help='Add task prompt')
    parser.add_argument('--sort_by_dict', action='store_true', help='Arrange elements by their label names')

    # tasks
    parser.add_argument('--tasks', type=str, default='refinement',
                        help='tasks to train')
    parser.add_argument('--task_weights', type=str, required=False,
                        help='sampling weights for each task')
    parser.add_argument('--task_loss_weights', type=str, required=False,
                        help='loss weights for each task')
    parser.add_argument('--enable_task_measure', action='store_true',
                        help='whether to compute mIoU for each task in validation')
    parser.add_argument('--partition_training_data', action='store_true',
                        help='whether to partition training data')
    parser.add_argument('--partition_training_data_task_buckets', type=str, required=False,
                        help='task buckets in partition_training_data')
    parser.add_argument('--fine_grained_partition_training_data', action='store_true',
                        help='whether to partition training data')
    parser.add_argument('--fine_grained_partition_training_data_task_size', type=str, required=False,
                        help='task data size in fine-grained partition_training_data')
    parser.add_argument('--single_task_per_batch', action='store_true', default=False,
                        help='Do not mix tasks in a batch. Perform a task per batch.')

    # Refinement
    parser.add_argument('--gaussian_noise_mean', type=float, default=0.0,
                        help='gaussian noise mean')
    parser.add_argument('--gaussian_noise_std', type=float, default=0.01,
                        help='gaussian noise std')
    parser.add_argument('--train_bernoulli_beta', type=float, default=1.0,
                        help='bernoulli beta in training, determining the probabaility of an element being corrupted')
    parser.add_argument('--test_bernoulli_beta', type=float, default=1.0,
                        help='bernoulli beta in testing, determining the probabaility of an element being corrupted')
    parser.add_argument('--refinement_shuffle_before_sort_by_label', action='store_true', default=False,
                        help='Shuffle elements before sorting by label in Refinement')
    parser.add_argument('--refinement_sort_by_pos_before_sort_by_label', action='store_true', default=False,
                        help='Sort elements by position before sorting by label in Refinement')

    # completion
    parser.add_argument('--completion_sort_by_pos', action='store_true', default=False,
                        help='Sort elements by position in completion')
    parser.add_argument('--completion_shuffle_before_sort_by_label', action='store_true', default=False,
                        help='Shuffle elements before sorting by label in Completion')
    parser.add_argument('--completion_sort_by_pos_before_sort_by_label', action='store_true', default=False,
                        help='Sort elements by position before sorting by label in Completion')

    # ugen
    parser.add_argument('--ugen_sort_by_pos', action='store_true', default=False,
                        help='Sort elements by position in UGen')
    parser.add_argument('--ugen_shuffle_before_sort_by_label', action='store_true', default=False,
                        help='Shuffle elements before sorting by label in UGen')
    parser.add_argument('--ugen_sort_by_pos_before_sort_by_label', action='store_true', default=False,
                        help='Sort elements by position before sorting by label in UGen')

    # gen_t
    parser.add_argument('--gen_t_add_unk_token', action='store_true', default=False,
                        help='Whether to add unk token in the position & size of elements')
    parser.add_argument('--gen_t_shuffle_before_sort_by_label', action='store_true', default=False,
                        help='Shuffle elements before sorting by label in Gen_T')
    parser.add_argument('--gen_t_sort_by_pos_before_sort_by_label', action='store_true', default=False,
                        help='Sort elements by position before sorting by label in Gen_T')

    # gen_ts
    parser.add_argument('--gen_ts_add_unk_token', action='store_true', default=False,
                        help='Whether to add unk token in the position of elements')
    parser.add_argument('--gen_ts_shuffle_before_sort_by_label', action='store_true', default=False,
                        help='Shuffle elements before sorting by label in Gen_TS')
    parser.add_argument('--gen_ts_sort_by_pos_before_sort_by_label', action='store_true', default=False,
                        help='Sort elements by position before sorting by label in Gen_TS')

    # gen_r
    parser.add_argument('--gen_r_discrete_before_induce_relations', action='store_true', default=False,
                        help='whether to perform discretization first')
    parser.add_argument('--gen_r_compact', action='store_true', default=False,
                        help='Make the input sequence of relation constrained compact')
    parser.add_argument('--gen_r_add_unk_token', action='store_true', default=False,
                        help='Whether to add unk token in the position of elements')
    parser.add_argument('--gen_r_shuffle_before_sort_by_label', action='store_true', default=False,
                        help='Shuffle elements before sorting by label in Gen_R')
    parser.add_argument('--gen_r_sort_by_pos_before_sort_by_label', action='store_true', default=False,
                        help='Sort elements by position before sorting by label in Gen_R')

    # Evaluation Random seed
    parser.add_argument('--eval_seed', type=int, required=True,
                        help='evaluation seed')
    parser.add_argument('--remove_too_long_layout', action='store_true', default=False,
                        help='whether to remove too long layout (len(seq) > 512)')

    parser.add_argument('--decode_max_length', default=350, type=int,
                        help='decode max length during evaluation')
    parser.add_argument('--topk', default=10, type=int,
                        help='parameter in topk sampling')
    parser.add_argument('--temperature', default=0.7, type=float,
                        help='parameter in topk sampling')
    parser.add_argument('--eval_interval', default=1, type=int,
                        help='evaluation interval during training')

    # Evaluate ckpt-tag
    parser.add_argument('--eval_ckpt_tag', type=str,
                        help='checkpoint tag')


def add_trainer_arguments(parser) -> None:
    parser.add_argument('--local_rank', type=int, help='distributed trainer', default=0)
    parser.add_argument('--trainer', type=str, choices=['basic', 'ddp', 'deepspeed'],
                        default='basic', help='trainer name')
    parser.add_argument('--backend', type=str, default='nccl', help='distributed backend')

    # Load training ckpt
    parser.add_argument('--load_train_ckpt', action='store_true', help='Load ckpt for training')
    parser.add_argument('--train_ckpt_path', type=str, help='train ckpt path')
    # Deepspeed checkpoint tag
    parser.add_argument('--ds_ckpt_tag', type=str, help='checkpoint tag', required=False)

    deepspeed.add_config_arguments(parser)
