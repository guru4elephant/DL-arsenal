import sys
import paddle.fluid as fluid
import paddle as paddle
from nn.linear import Linear
from nn.loss import SoftmaxWithCrossEntropy
from nn.loss import Softmax
from nn.loss import CrossEntropy
from nn.math import Mean
from nn.data import ImageData
from nn.memory import GlobalMemory
import numpy as np

if len(sys.argv) != 3:
    print >> sys.stderr, "python run_from_binary_program.py startup.program main.program"
    sys.exit(-1)

startup_program = None
main_program = None

with open(sys.argv[1], "rb") as f:
    program_desc_str = f.read()
    startup_program = fluid.Program.parse_from_string(program_desc_str)

with open(sys.argv[2], "rb") as f:
    program_desc_str = f.read()
    main_program = fluid.Program.parse_from_string(program_desc_str)

memory = GlobalMemory(startup_program, main_program)

with memory.hold():
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(memory.startup_program)
    print("run startup program succeed")
    train_reader = paddle.batch(
        paddle.dataset.mnist.train(), batch_size=16)
    
    for i in range(10):
        for j, data in enumerate(train_reader()):
            img_data = np.array([x[0].reshape([1, 784]) for x in data]).astype('float32')
            y_data = np.array([x[1] for x in data]).astype("int64")
            print y_data
            y_data = y_data.reshape([len(y_data), 1])
            exe.run(memory.main_program, 
                    feed = {image.name: img_data,
                            label.name: y_data},
                    fetch_list = [loss.name])
            print "run a mini-batch"


