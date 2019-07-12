import paddle.fluid as fluid
import numpy as numpy

x = fluid.layers.data(name="x", dtype="int64", shape=[1], lod_level=1)
w = fluid.layers.data(name="w", dtype="float32", shape=[1], lod_level=1)

emb = fluid.layers.embedding(input=x, size=[10000, 10])
w = fluid.layers.expand(x=w, expand_times=[1, 10])

rescaled_emb = fluid.layers.elementwise_mul(x=emb, y=w)

exe = fluid.Executor(place=fluid.CPUPlace())
exe.run(fluid.default_startup_program())
rescale_emb_np = exe.run(feed={
    "x": fluid.create_lod_tensor(data=numpy.array([1, 3, 4, 5, 3, 6, 8], dtype='int64').reshape(-1, 1),
                                 recursive_seq_lens=[[4, 1, 2]],
                                 place=fluid.CPUPlace()),
    "w": fluid.create_lod_tensor(data=numpy.array([0.1, 0.3, 0.4, 0.5, 0.3, 0.6, 0.8], dtype='float32').reshape(-1, 1),
                                 recursive_seq_lens=[[4, 1, 2]],
                                 place=fluid.CPUPlace())},
                         fetch_list=[rescaled_emb], return_numpy=False)

