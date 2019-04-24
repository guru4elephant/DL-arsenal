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
import paddle
import logging
import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.collective as fleet


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

os.environ["NCCL_SOCKET_IFNAME"] = "xgbe0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_CUDA_SUPPORT"] = "1"
os.environ["NCCL_IB_DISABLE"] = "0"

def load_vocab(filename):
    vocab = {}
    with open(filename) as f:
        wid = 0
        for line in f:
            vocab[line.strip()] = wid
            wid += 1
    vocab["<unk>"] = len(vocab)
    return vocab

fleet.init()

if __name__ == "__main__":
    vocab = load_vocab('imdb.vocab')
    dict_dim = len(vocab)

    data = fluid.layers.data(name="words", shape=[1], dtype="int64", lod_level=1)
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
    filelist = ["train_data/%s" % x for x in os.listdir("train_data")]
    dataset.set_use_var([data, label])
    pipe_command = "python imdb_reader.py"
    dataset.set_pipe_command(pipe_command)
    dataset.set_batch_size(64)
    dataset.set_filelist(filelist)
    dataset.set_thread(1)
    dataset.load_into_memory()
    from nets import lstm_net
    avg_cost, acc, prediction = lstm_net(data, label, dict_dim)
    sgd = fluid.optimizer.SGD(learning_rate=0.01)
    sgd = fleet.DistributedOptimizer(sgd)
    sgd.minimize(avg_cost)
    fleet.init_worker(fluid.default_main_program())
    exe = fluid.Executor(fluid.CUDAPlace(fleet.worker_index()))
    exe.run(fluid.default_startup_program())
    epochs = 30
    save_dirname = "lstm_model"
    for i in range(epochs):
        exe.train_from_dataset(program=fluid.default_main_program(),
                               fetch_list=[avg_cost],
                               fetch_info=["avg_cost"],
                               dataset=dataset, debug=False)
        logger.info("TRAIN --> pass: {}".format(i))
        fluid.io.save_inference_model("%s/epoch%d.model" % (save_dirname, i),
                                      [data.name, label.name], [acc], exe)
