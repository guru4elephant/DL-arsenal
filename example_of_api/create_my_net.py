import paddle.fluid as fluid
from nn.linear import Linear
from nn.cross_entropy import SoftmaxWithCrossEntropy
from nn.data import ImageData

my_program = fluid.Program()
my_block = my_program.current_block()

class ClassifyNet():
    def __init__(self):
        self.data = ImageData(my_block, "data", 768)
        self.fc1_layer = Linear(my_block, "fc1", 768, 256)
        self.fc2_layer = Linear(my_block, "fc2", 256, 64)
        self.fc3_layer = Linear(my_block, "fc3", 64, 2)
        self.loss_layer = SoftmaxWithCrossEntropy(my_block, "cross_entropy")

class_net = ClassifyNet()
image, label = class_net.data()
fc1 = class_net.fc1_layer(image)
fc2 = class_net.fc2_layer(fc1)
fc3 = class_net.fc3_layer(fc2)
loss = class_net.loss_layer(fc3, label)

adagrad_opt = fluid.optimizer.Adagrad(learning_rate=0.1)
adagrad_opt.minimize(loss)
print(my_program)


