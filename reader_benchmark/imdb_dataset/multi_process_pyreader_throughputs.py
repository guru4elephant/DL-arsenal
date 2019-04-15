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

def pyreader_only(data, label):
    py_reader = fluid.layers.create_py_reader_by_data(
        capacity=128, feed_list=[data, label], name='py_reader',
        use_double_buffer=True)
    inputs = fluid.layers.read_file(py_reader)
    return py_reader

if __name__ == "__main__":
    batch = int(sys.argv[1])
    thread = int(sys.argv[2])
    os.environ["CPU_NUM"] = str(thread)

    data = fluid.layers.data(name="words", shape=[1], dtype="int64", lod_level=1)
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    py_reader = pyreader_only(data, label)
    from imdb_reader import IMDBDataset
    dataset = IMDBDataset()
    dataset.init_hashing()
    filelist = ["train_data/part-0"]
    
    readers = []
    for i in range(thread):
        readers.append(dataset.infer_reader(filelist, batch, 1000))

    multi_readers = paddle.reader.multiprocess_reader(readers)

    py_reader.decorate_paddle_reader(multi_readers)
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    build_strategy = fluid.BuildStrategy()
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = thread
    train_exe = fluid.ParallelExecutor(
        use_cuda=False,
        main_program=fluid.default_main_program(),
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)

    epochs = 3
    batch_id = 0
    for i in range(epochs):
        py_reader.start()
        try:
            while True:
                train_exe.run(fetch_list=[])
                batch_id += 1
        except fluid.core.EOFException:
            logger.info(
                "TRAIN --> pass: {}".
                format(i + 1))
            py_reader.reset()
