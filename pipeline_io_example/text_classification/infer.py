#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time
import unittest
import contextlib
import numpy as np

import paddle
import paddle.fluid as fluid
import logging

from imdb_reader import IMDBDataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res

def data2tensor(data, place):
    input_seq = to_lodtensor([x[0] for x in data], place)
    y_data = np.array([x[1] for x in data]).astype("int64")
    y_data = y_data.reshape([-1, 1])
    return {"words": input_seq, "label": y_data}

def infer(test_reader, use_cuda, model_path=None):
    """
    inference function
    """
    if model_path is None:
        print(str(model_path) + " cannot be found")
        return
    
    
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(model_path, exe)

        total_acc = 0.0
        total_count = 0
        for data in test_reader():
            acc = exe.run(inference_program,
                          feed=data2tensor(data, place),
                          fetch_list=fetch_targets,
                          return_numpy=True)
            total_acc += acc[0] * len(data)
            total_count += len(data)

        avg_acc = total_acc / total_count
        logger.info("Model Path: {}, avg_acc: {}".format(model_path, avg_acc))


if __name__ == "__main__":
    if __package__ is None:
        from os import sys, path
        sys.path.append(
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    batch_size = 128
    model_path = sys.argv[1]

    if len(sys.argv) == 3:
        test_data_dirname = sys.argv[2]
    imdb = IMDBDataset()
    imdb.load_resource("imdb.vocab")
    test_reader = imdb.infer_reader(["test_data/%s" % x for x in os.listdir("test_data")], 128, 10000)
    models = os.listdir(model_path)
    for i in range(0, len(models)):
        epoch_path = "epoch" + str(i) + ".model"
        epoch_path = os.path.join(model_path, epoch_path)
        infer(test_reader, use_cuda=False, model_path=epoch_path)
