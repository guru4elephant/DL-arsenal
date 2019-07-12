import paddle.fluid as fluid
import numpy as numpy
sentence = fluid.layers.data(name="sentence", dtype="int64", shape=[1], lod_level=1)

exe = fluid.Executor(place=fluid.CPUPlace())
exe.run(feed={
    "sentence": fluid.create_lod_tensor(
        data=numpy.array([1, 3, 4, 5, 3, 6, 8], dtype='int64').reshape(-1, 1),
        recursive_seq_lens=[[4, 1, 2]],
        place=fluid.CPUPlace()
    )
})
