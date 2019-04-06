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
from nets import bow_encoder
base_lr = 0.0001
batch_size = 128
emb_lr = 5.0 * batch_size
fc_lr = 200.0
dict_dim = 1451594
emb_dim = 128
hid_dim = 128
margin = 0.1


q = fluid.layers.data(
    name="query", shape=[1], dtype="int64", lod_level=1)
pt = fluid.layers.data(
    name="pos_title", shape=[1], dtype="int64", lod_level=1)
nt = fluid.layers.data(
    name="neg_title", shape=[1], dtype="int64", lod_level=1)

avg_cost, pt_s, nt_s, pnum, nnum, train_pn = \
        bow_encoder(q, pt, nt, dict_dim, emb_dim,
                    hid_dim, emb_lr, fc_lr, margin)

sgd_optimizer = fluid.optimizer.SGD(learning_rate=base_lr)
sgd_optimizer.minimize(avg_cost)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

thread_num = 1
dataset = fluid.DatasetFactory().create_dataset()
dataset.set_batch_size(batch_size)
dataset.set_use_var([q, pt, nt])
dataset.set_batch_size(batch_size)
pipe_command = 'python pairwise_file_reader.py'
dataset.set_pipe_command(pipe_command)
filelist = ["train_raw/%s" % x for x in os.listdir("train_raw")]

dataset.set_filelist(filelist[:int(0.9*len(filelist))])
dataset.set_thread(thread_num)
epochs = 40

save_dirname = "simnet_bow_model"
for i in range(epochs):
    dataset.set_filelist(filelist[:int(0.9*len(filelist))])
    exe.train_from_dataset(program=fluid.default_main_program(),
                           dataset=dataset,
                           fetch_list=[train_pn],
                           fetch_info=["pos/neg"],
                           print_period=10000,
                           debug=False)
    sys.stderr.write("epoch%d finished" % (i + 1))
    fluid.io.save_inference_model("%s/epoch%d.model" % (save_dirname, (i + 1)),
                                  [q.name, pt.name, nt.name], [pnum, nnum], exe)

