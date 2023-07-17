def add_arguments(parser):

    # data dir
    parser.add_argument('--data_dir', type=str, default='../datasets/',
                        help='data dir')
    parser.add_argument('--out_dir', type=str, default='../output/fid/',
                        help='output directory')

    # dataset
    parser.add_argument('--dataset', type=str, choices=['rico', 'publaynet'],
                        help='dataset name')
    parser.add_argument('--num_label', type=int, default=25,
                        help='number of labels')
    parser.add_argument('--max_num_elements', type=int, default=20,
                        help='max number of design elements')

    # hyperparameters
    parser.add_argument('--epoch', type=int, default=50, help='number of epoch to train')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size for each GPU')
    parser.add_argument('--train_log_step', type=int, default=50,
                        help='number of steps to log loss during training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')

    parser.add_argument('--gaussian_noise_mean', type=float, default=0.0,
                        help='gaussian noise mean')
    parser.add_argument('--gaussian_noise_std', type=float, default=0.05,
                        help='gaussian noise std')
    parser.add_argument('--train_bernoulli_beta', type=float, default=1.0,
                        help='bernoulli beta in training, determining the probabaility of an element being corrupted')

    # loss weight
    parser.add_argument('--box_loss_weight', type=float, default=10.0,
                        help='box loss weight')
