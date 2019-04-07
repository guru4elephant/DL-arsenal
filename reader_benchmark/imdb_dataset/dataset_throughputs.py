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

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    batch = int(sys.argv[1])
    thread = int(sys.argv[2])

    data = fluid.layers.data(name="words", shape=[1], dtype="int64", lod_level=1)
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var([data, label])
    dataset.set_pipe_command("python imdb_reader.py")
    dataset.set_batch_size(batch)

    filelist = ["train_data/part.all"] * thread
    dataset.set_filelist(filelist)
    dataset.set_thread(thread)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    epochs = 1
    for i in range(epochs):
        exe.train_from_dataset(program=fluid.default_main_program(),
                               dataset=dataset, debug=True)
        logger.info("TRAIN --> pass: {}".format(i + 1))
