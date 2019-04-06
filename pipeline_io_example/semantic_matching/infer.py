#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import numpy as np
from pairwise_file_reader import PairwiseReader

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
    query = to_lodtensor([x[0] for x in data], place)
    pos_title = to_lodtensor([x[1] for x in data], place)
    neg_title = to_lodtensor([x[2] for x in data], place)
    return {"query": query, "pos_title": pos_title, "neg_title": neg_title}

def infer_from_dataset(model_path=None):
    if model_path is None:
        print(str(model_path) + "cannot be found")
        return
    
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets] \
            = fluid.io.load_inference_model(model_path, exe)
        feed_vars = []
        for name in feed_target_names:
            feed_vars.append(inference_program.global_block().var(name))
        fetch_vars = []
        for ft in fetch_targets:
            fetch_vars.append(inference_program.global_block().var(ft.name.replace("save_inference_model\/", "")))
        with open("inference_program.prog.txt", "w") as fout:
            fout.write(str(inference_program))
        dataset = fluid.DatasetFactory().create_dataset()
        dataset.set_batch_size(128)
        dataset.set_use_var(feed_vars)
        pipe_command = '/home/users/dongdaxiang/paddle_whls/pipe_reader/paddle_release_home/python/bin/python pairwise_file_reader.py'
        dataset.set_pipe_command(pipe_command)
        filelist = ["train_raw/%s" % x for x in os.listdir("train_raw")]
        dataset.set_filelist(filelist[int(0.9*len(whole_filelist)):int(0.9*len(whole_filelist))+2])
        dataset.set_thread(1)
        exe.infer_from_dataset(program=inference_program,
                               dataset=dataset,
                               fetch_list=fetch_vars,
                               fetch_info=["pos_num", "neg_num"],
                               print_period=1,
                               debug=False)


def infer(test_reader, model_path=None):
    if model_path is None:
        print(str(model_path) + "cannot be found")
        return

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets] \
            = fluid.io.load_inference_model(model_path, exe)

        total_pnum = 0.0
        total_nnum = 0.0
        for i, data in enumerate(test_reader()):
            pnum, nnum = exe.run(inference_program,
                                 feed=data2tensor(data, place),
                                 fetch_list=fetch_targets)
            total_pnum += pnum[0]
            total_nnum += nnum[0]
            if i > 0 and i % 1000 == 0:
                print(total_pnum / total_nnum)
        print("%s\t%f" % (model_path, total_pnum / total_nnum))

if __name__ == "__main__":
    if __package__ is None:
        from os import sys, path
        sys.path.append(
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    model_path = sys.argv[1]
    dataset = PairwiseReader()
    whole_filelist = ["train_raw/%s" % x for x in os.listdir("train_raw")]

    test_reader = dataset.infer_reader(whole_filelist[int(0.9*len(whole_filelist)) - 1:int(0.9*len(whole_filelist))], 128, 10000)
    models = os.listdir(model_path)
    epoch = int(sys.argv[2])
    epoch_path = "epoch" + str(epoch) + ".model"
    epoch_path = os.path.join(model_path, epoch_path)
    #infer(test_reader, model_path=epoch_path)
    infer_from_dataset(model_path=epoch_path)
    
