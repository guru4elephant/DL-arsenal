#!/usr/bin/python
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import math
import paddle.fluid as fluid
from argument import params_args

feature_names = []
with open(sys.argv[1]) as fin:
    for line in fin:
        feature_names.append(line.strip())

sparse_input_ids = [
    fluid.layers.data(name=name, shape=[1], lod_level=1, dtype='int64')
    for name in feature_names]

label = fluid.layers.data(
    name='label', shape=[1], dtype='int64')

sparse_feature_dim = 1000001
embedding_size = 9

def embedding_layer(input):
    emb = fluid.layers.embedding(
        input=input,
        is_sparse=True,
        is_distributed=False,
        size=[sparse_feature_dim, embedding_size],
        param_attr=fluid.ParamAttr(name="SparseFeatFactors",
                                   initializer=fluid.initializer.Uniform()))
    emb_sum = fluid.layers.sequence_pool(input=emb, pool_type='sum')
    return emb_sum

emb_sums = list(map(embedding_layer, sparse_input_ids))
concated = fluid.layers.concat(emb_sums, axis=1)
fc1 = fluid.layers.fc(input=concated, size=400, act='relu',
                      param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                          scale=1 / math.sqrt(concated.shape[1]))))
fc2 = fluid.layers.fc(input=fc1, size=400, act='relu',
                      param_attr=fluid.ParamAttr(
                          initializer=fluid.initializer.Normal(
                              scale=1 / math.sqrt(fc1.shape[1]))))
fc3 = fluid.layers.fc(input=fc2, size=400, act='relu',
                      param_attr=fluid.ParamAttr(
                          initializer=fluid.initializer.Normal(
                              scale=1 / math.sqrt(fc2.shape[1]))))

predict = fluid.layers.fc(input=fc3, size=2, act='softmax',
                          param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                              scale=1 / math.sqrt(fc3.shape[1]))))

cost = fluid.layers.cross_entropy(input=predict, label=words[-1])
avg_cost = fluid.layers.reduce_sum(cost)
accuracy = fluid.layers.accuracy(input=predict, label=words[-1])
auc_var, batch_auc_var, auc_states = \
    fluid.layers.auc(input=predict, label=words[-1], num_thresholds=2 ** 12, slide_steps=20)

dataset = fluid.DatasetFactory().create_dataset()
dataset.set_use_var(self.sparse_input_ids + [label])
pipe_command = "python dataset_generator.py"
dataset.set_pipe_command(pipe_command)
dataset.set_batch_size(params.batch_size)
thread_num = int(params.cpu_num)
dataset.set_thread(thread_num)

optimizer = fluid.optimizer.SGD(params.learning_rate)
optimizer.minimize(loss)
exe = fluid.Executor(fluid.CPUPlace())

train_folder = ["afs:/app/fs/20191020", "afs:/app/fs/20191021"]
train_filelists = [["afs:/app/fs/20191020/0.txt", "afs:/app/fs/20191020/1.txt"],
                   ["afs:/app/fs/20191021/0.txt", "afs:/app/fs/20191021/1.txt"]]

exe.run(fluid.default_startup_program())
for filelist in train_filelists:
    dataset.set_filelist(filelist)
    exe.train_from_dataset(
        program=fluid.default_main_program(),
        dataset=dataset,
        fetch_list=[auc_var],
        fetch_info=["auc"],
        debug=False)
    # save model here

