from __future__ import print_function
import argparse
import logging
import os
import sys
import time
import math
import random
import numpy as np
import paddle
import paddle.fluid as fluid
import six
import reader
from net import skip_gram_word2vec, skip_gram_word2vec_dataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(
        description="PaddlePaddle Word2vec example")
    parser.add_argument(
        '--train_data_dir',
        type=str,
        default='./data/text',
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
        default='./data/1-billion_dict',
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
    return parser.parse_args()

def GetFileList(data_path):
    return [data_path + "/" + x for x in os.listdir(data_path)]

def train(args):

    if not os.path.isdir(args.model_output_dir):
        os.mkdir(args.model_output_dir)

    logger.info("dict_size: {}".format(word2vec_reader.dict_size))

    input_word = fluid.layers.data(
        name="context_id", shape=[1], dtype="int64", lod_level=0)
    true_word = fluid.layers.data(
        name="target", shape=[1], dtype="int64", lod_level=0)
    neg_num = 5
    neg_word = fluid.layers.data(
        name="neg_label", shape=[neg_num], dtype='int64', lod_level=0)

    loss = skip_gram_word2vec_dataset(input_word,
                                      true_word,
                                      neg_word,
                                      word2vec_reader.dict_size,
                                      id_frequencys_pow,
                                      args.embedding_size,
                                      is_sparse=args.is_sparse)

    optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.layers.exponential_decay(
                learning_rate=args.base_lr,
                decay_steps=100000,
                decay_rate=0.999,
                staircase=True))

    optimizer.minimize(loss)

    # do local training 
    logger.info("run local training")
    main_program = fluid.default_main_program()

    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var([input_word, true_word, neg_word])
    dataset.set_batch_size(args.batch_size)
    dataset.set_pipe_command("python new_reader.py")
    dataset.set_thread(40)
    filelist = GetFileList(args.train_data_dir) * 40
    dataset.set_filelist(filelist)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    for i in range(args.num_passes):
        exe.train_from_dataset(
            program=fluid.default_main_program(),
            dataset=dataset, debug=True)

if __name__ == '__main__':
    args = parse_args()
    train(args)
