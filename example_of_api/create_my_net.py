import paddle.fluid as fluid
from nn.linear import Linear
from nn.cross_entropy import SoftmaxWithCrossEntropy
from nn.data import ImageData
from nn.memory import GlobalMemory

class ClassifyNet():
    def __init__(self, memory):
        self.data = ImageData(memory, "data", 768)
        self.fc1_layer = Linear(memory, "fc1", 768, 256)
        self.fc2_layer = Linear(memory, "fc2", 256, 64)
        self.fc3_layer = Linear(memory, "fc3", 64, 2)
        self.loss_layer = SoftmaxWithCrossEntropy(memory, "cross_entropy")

memory = GlobalMemory()

with memory.hold():
    class_net = ClassifyNet(memory)
    image, label = class_net.data()
    fc1 = class_net.fc1_layer(image)
    fc2 = class_net.fc2_layer(fc1)
    fc3 = class_net.fc3_layer(fc2)
    loss = class_net.loss_layer(fc3, label)
    
    adagrad_opt = fluid.optimizer.Adagrad(learning_rate=0.1)
    adagrad_opt.minimize(loss, startup_program=memory.startup_program)
    print(loss.block.program)


