from __future__ import print_function
import argparse
import logging
import os
import time

import numpy as np

# disable gpu training for this example
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import paddle
import paddle.fluid as fluid
from paddle.fluid.executor import global_scope

import reader
from network_conf import skip_gram_word2vec
from infer import inference_test

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PaddlePaddle Word2vec example")
    parser.add_argument(
        '--train_data_path',
        type=str,
        default='./data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled',
        help="The path of training dataset")
    parser.add_argument(
        '--dict_path',
        type=str,
        default='./data/1-billion_dict',
        help="The path of data dict")
    parser.add_argument(
        '--test_data_path',
        type=str,
        default='./data/text8',
        help="The path of testing dataset")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help="The size of mini-batch (default:100)")
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help="epoch of training")
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
        '--with_hs',
        action='store_true',
        required=False,
        default=False,
        help='using hierarchical sigmoid, (default: False)')

    parser.add_argument(
        '--with_nce',
        action='store_true',
        required=False,
        default=False,
        help='using negtive sampling, (default: True)')

    parser.add_argument(
        '--max_code_length',
        type=int,
        default=40,
        help='max code length used by hierarchical sigmoid, (default: 40)')

    parser.add_argument(
        '--is_sparse',
        action='store_true',
        required=False,
        default=False,
        help='embedding and nce will use sparse or not, (default: False)')

    parser.add_argument(
        '--with_Adam',
        action='store_true',
        required=False,
        default=False,
        help='Using Adam as optimizer or not, (default: False)')

    parser.add_argument(
        '--is_local',
        action='store_true',
        required=False,
        default=False,
        help='Local train or not, (default: False)')

    parser.add_argument(
        '--with_speed',
        action='store_true',
        required=False,
        default=False,
        help='print speed or not , (default: False)')

    parser.add_argument(
        '--with_infer_test',
        action='store_true',
        required=False,
        default=False,
        help='Do inference every 100 batches , (default: False)')

    parser.add_argument(
        '--rank_num',
        type=int,
        default=4,
        help="find rank_num-nearest result for test (default: 4)")

    parser.add_argument(
        '--use_pyreader',
        required=False,
        default=False,
        help='Whether you want to use pyreader, (default: False)')
    return parser.parse_args()


def async_train_loop(args, train_program, loss, dataset, filelist):
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    async_executor = fluid.AsyncExecutor(place)
    thread_num = 40
    for i in range(args.epochs):
        async_executor.run(
            train_program, # main program
            dataset, # dataset
            filelist, # filelist
            thread_num, # thread
            [], # fetch
            debug=True) # debug
        epoch_model = "word2vec_model/epoch" + str(i + 1)
        fluid.io.save_inference_model(
            epoch_model,
            [data.name, label.name],
            [acc],
            executor)

def GetFileList(data_path):
    #return data_path + "/" + os.listdir(data_path)
    res_list = [data_path + "/" + x for x in os.listdir(data_path)]
    return res_list

def async_train(args):
    if not os.path.isdir(args.model_output_dir):
                os.mkdir(args.model_output_dir)
    filelist = GetFileList(args.train_data_path)
    word2vec_reader = reader.Word2VecReader(
        args.dict_path, args.train_data_path, filelist, 0, 1)
    loss, words = skip_gram_word2vec(
        word2vec_reader.dict_size,
        word2vec_reader.word_frequencys,
        args.embedding_size,
        args.max_code_length,
        args.with_hs,
        args.with_nce,
        is_sparse=args.is_sparse)
    dataset = fluid.DataFeedDesc('data_feed.proto')
    dataset.set_batch_size(args.batch_size)
    dataset.set_use_slots([w.name for w in words])
    dataset.set_pipe_command("/home/users/dongdaxiang/paddle_whls/new_io/paddle_release_home/python/bin/python word2vec_data_gen.py")
    optimizer = fluid.optimizer.SGD(learning_rate=1e-4)
    optimizer.minimize(loss)
    async_train_loop(args, fluid.default_main_program(), loss, dataset, filelist)


if __name__ == '__main__':
    args = parse_args()
    async_train(args)
