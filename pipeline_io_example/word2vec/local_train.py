from __future__ import print_function
from args import parse_args
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

from net import skip_gram_word2vec_dataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

def GetFileList(data_path):
    return [data_path + "/" + x for x in os.listdir(data_path)]

def train(args):
    if not os.path.isdir(args.model_output_dir):
        os.mkdir(args.model_output_dir)

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
                                      354052,
                                      None,
                                      args.embedding_size,
                                      is_sparse=args.is_sparse)
    optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.layers.exponential_decay(
                learning_rate=args.base_lr,
                decay_steps=100000,
                decay_rate=0.999,
                staircase=True))

    optimizer.minimize(loss)

    main_program = fluid.default_main_program()

    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var([input_word, true_word, neg_word])
    dataset.set_pipe_command("sudo /home/users/dongdaxiang/paddle_whls/pipe_reader/paddle_release_home/python/bin/python reader.py")
    dataset.set_batch_size(args.batch_size)
    filelist = GetFileList(args.train_data_dir)
    dataset.set_filelist(filelist)
    dataset.set_thread(args.thread_num)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    for i in range(args.epochs):
        logger.info("Going to train epoch {}".format(i))
        exe.train_from_dataset(program=fluid.default_main_program(),
                               dataset=dataset)

if __name__ == '__main__':
    args = parse_args()
    train(args)
