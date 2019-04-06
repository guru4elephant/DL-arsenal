import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="PaddlePaddle Word2vec example")
    parser.add_argument(
        '--train_data_dir',
        type=str,
        default='./convert_1billion',
        help="The path of taining dataset")
    parser.add_argument(
        '--base_lr',
        type=float,
        default=0.01,
        help="The number of learing rate (default: 0.01)")
    parser.add_argument(
        '--save_step',
        type=int,
        default=500000,
        help="The number of step to save (default: 500000)")
    parser.add_argument(
        '--print_batch',
        type=int,
        default=10,
        help="The number of print_batch (default: 10)")
    parser.add_argument(
        '--dict_path',
        type=str,
        default='./data/new_1-billion-dict',
        help="The path of data dict")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=500,
        help="The size of mini-batch (default:500)")
    parser.add_argument(
        '--num_passes',
        type=int,
        default=10,
        help="The number of passes to train (default: 10)")
    parser.add_argument(
        '--model_output_dir',
        type=str,
        default='models',
        help='The path for model to store (default: models)')
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=64,
        help='sparse feature hashing space for index processing')
    parser.add_argument(
        '--is_sparse',
        action='store_true',
        required=False,
        default=False,
        help='embedding and nce will use sparse or not, (default: False)')
    parser.add_argument(
        '--with_speed',
        action='store_true',
        required=False,
        default=False,
        help='print speed or not , (default: False)')
    parser.add_argument(
        '--thread_num',
        type=int,
        default=10,
        help='thread number for training')
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='number of epochs for training')
    return parser.parse_args()
