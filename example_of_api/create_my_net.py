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

class ClassifyNet():
    def __init__(self, memory):
        self.data = ImageData(memory, "data", 784)
        self.fc1_layer = Linear(memory, "fc1", 784, 256)
        self.fc2_layer = Linear(memory, "fc2", 256, 64)
        self.fc3_layer = Linear(memory, "fc3", 64, 10)
        self.softmax = Softmax(memory, "softmax")
        self.cross_entropy = CrossEntropy(memory, "entropy")
        self.mean = Mean(memory, "mean")

memory = GlobalMemory()

with memory.hold():
    class_net = ClassifyNet(memory)
    image, label = class_net.data()
    fc1 = class_net.fc1_layer(image)
    fc2 = class_net.fc2_layer(fc1)
    fc3 = class_net.fc3_layer(fc2)
    softmax = class_net.softmax(fc3)
    loss = class_net.cross_entropy(softmax, label)
    mean_loss = class_net.mean(loss)
    adagrad_opt = fluid.optimizer.Adagrad(learning_rate=0.1)
    adagrad_opt.minimize(mean_loss, startup_program=memory.startup_program)
    print("total weight parameter num: %d" % memory.weight_param_num)
    print("total variable parameter num: %d" % memory.var_param_num)
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


