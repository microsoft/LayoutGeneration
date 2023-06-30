import os


def add_arguments(parser):

    parser.add_argument('--test', dest='test', action='store_true',
                        help='whether to do inference')
    parser.add_argument('--train', dest='test', action='store_false',
                        help='whether to do training')

    # data dir
    parser.add_argument('--data_dir', type=str, default=os.getenv('AMLT_DATA_DIR', '../datasets/'),
                        help='data dir')
    parser.add_argument('--out_dir', type=str, default=os.getenv('AMLT_OUTPUT_DIR', '../output/'),
                        help='output directory')

    # data
    parser.add_argument('--dataset', type=str, choices=['rico', 'publaynet'],
                        help='dataset name')
    parser.add_argument('--max_num_elements', type=int, default=20,
                        help='max number of design elements')

    # pretrain LM
    parser.add_argument('--pretrained_lm_name', default='t5-small', type=str,
                        help='pretrain language model name')
    parser.add_argument('--with_pretrained_weights', dest='with_pretrained_weights', 
                        action='store_true', default=True, 
                        help='Load pretrained weights')
    parser.add_argument('--without_pretrained_weights', dest='with_pretrained_weights',
                        action='store_false', help='Do not load pretrained weights')

    parser.add_argument('--epoch', type=int, default=30, help='number of epoch to train')
    parser.add_argument('--adafactor', action='store_true', default=False,
                        help='whether to use adafactor optimizer')

    parser.add_argument('--load_train_ckpt', action='store_true', default=False,
                        help='whether to load train checkpoint')

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

    # Data Format
    parser.add_argument('--shuffle_elements', action='store_true', default=False,
                        help='whehter to randomly shuffle elements')  # shuffle once is ok
    parser.add_argument('--bbox_format', type=str, choices=['xywh', 'ltrb', 'ltwh'], default='ltwh',
                        help='bbox coordinate ')
    parser.add_argument('--discrete', action='store_true', default=False,
                        help='discretize bounding box')
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
