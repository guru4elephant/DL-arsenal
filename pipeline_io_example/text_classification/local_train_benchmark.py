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
from args import parse_args

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

def load_vocab(filename):
    vocab = {}
    with open(filename) as f:
        wid = 0
        for line in f:
            vocab[line.strip()] = wid
            wid += 1
    vocab["<unk>"] = len(vocab)
    return vocab

if __name__ == "__main__":
    args = parse_args()
    if not os.path.isdir(args.model_output_dir):
        os.mkdir(args.model_output_dir)
    vocab = load_vocab('imdb.vocab')
    dict_dim = len(vocab)

    data = fluid.layers.data(name="words", shape=[1], dtype="int64", lod_level=1)
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    dataset = fluid.DatasetFactory().create_dataset(args.dataset_mode)
    filelist = ["train_data/%s" % x for x in os.listdir("train_data")]
    filelist = [filelist[0]] * 10
    dataset.set_use_var([data, label])
    pipe_command = "python imdb_reader.py"
    dataset.set_pipe_command(pipe_command)
    batch = 4
    from nets import *
    if args.text_encoder == "bow":
        network = bow_net
        batch = 128
    elif args.text_encoder == "cnn":
        network = cnn_net
    elif args.text_encoder == "gru":
        network = gru_net
    else:
        network = lstm_net

    dataset.set_batch_size(batch)
    dataset.set_filelist(filelist)
    dataset.set_thread(args.thread)
    if args.dataset_mode == "InMemoryDataset":
        dataset.load_into_memory()

    avg_cost, acc, prediction = network(data, label, dict_dim)
    optimizer = fluid.optimizer.SGD(learning_rate=0.01)
    optimizer.minimize(avg_cost)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    epochs = args.num_passes
    save_dirname = args.model_output_dir
    for i in range(epochs):
        exe.train_from_dataset(program=fluid.default_main_program(),
                               dataset=dataset, debug=False)
        logger.info("TRAIN --> pass: {}".format(i))
        fluid.io.save_inference_model("%s/epoch%d.model" % (save_dirname, i),
                                      [data.name, label.name], [acc], exe)


