import sys
import paddle.fluid as fluid
import paddle as paddle
from nn.linear import Linear
from nn.loss import SoftmaxWithCrossEntropy
from nn.loss import Softmax
from nn.loss import CrossEntropy
from nn.conv import Conv2d
from nn.pool import Pool2d
from nn.math import BatchNorm
from nn.math import Mean
from nn.math import ElemAdd
from nn.activation import Relu
from nn.data import ImageData
from nn.memory import GlobalMemory
from nn.module import Module
import numpy as np


class BottleneckBlock(Module):
    def __init__(self, memory, basename,
                 in_channel, out_channel, stride=1):
        super(BottleneckBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.basename = basename
        self.memory = memory
        self.conv0 = ConvBnNet(memory, "%s_conv0" % basename, in_channel, out_channel, 1, stride=1)
        self.conv1 = ConvBnNet(memory, "%s_conv1" % basename, out_channel, out_channel, 3, stride=stride, act='relu')
        self.conv2 = ConvBnNet(memory, "%s_conv2" % basename, out_channel, out_channel * 4, 1, stride=1, act='relu')
        if stride != 1 or in_channel != out_channel * 4:
            self.conv3 = ConvBnNet(memory, "%s_conv3" % basename, out_channel * 4, out_channel, 1, stride)
        self.add = ElemAdd(memory, "%s_shortcut" % basename)

    def forward(self, input):
        conv0 = self.conv0(input)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        if self.stride != 1 or self.in_channel != self.out_channel * 4:
            conv3 = self.conv3(conv2)
            result = self.add(conv3)
            return result
        return self.add(conv2)


class ConvBnNet(Module):
    def __init__(self, memory, basename, in_channel, out_channel, 
                 kernel_size, stride, padding=1, dilation=1,
                 groups=1, act='relu'):
        super(ConvBnNet, self).__init__()
        padding = (out_channel - 1) // 2
        self.act = act
        self.conv = Conv2d(memory, "%s_conv" % basename,
                           in_channel, out_channel,
                           kernel_size, stride=stride,
                           padding=padding, dilation=dilation,
                           groups=groups)
        self.bn = BatchNorm(memory, "%s_bn" % basename, out_channel)
        if act == 'relu':
            self.relu = Relu(memory, '%s_relu' % basename)
    def forward(self, input):
        conv = self.conv(input)
        bn = self.bn(conv)
        if self.act == 'relu':
            return self.relu(bn)
        return bn


memory = GlobalMemory()
data = ImageData(memory, "data", [3, 224, 224])

depth = [3, 4, 6, 3]
channels = [64, 64, 128, 256, 512]
conv1 = ConvBnNet(memory, "first_conv_bn", 3, channels[0], 
                  7, 2)
relu1 = Relu(memory, "first_relu")
pool1 = Pool2d(memory, "first_pool")

bottleneck_list = []
for block in range(len(depth)):
    for i in range(depth[block]):
        stride = 2 if i == 0 and block != 0 else 1
        bottle = BottleneckBlock(memory, "bottleneck_%d_%d" % (block, i), 
                                 channels[block], channels[block+1], stride=stride)
        bottleneck_list.append(bottle)

pool2 = Pool2d(memory, "second_pool")
fc1 = Linear(memory, "fc1", 12300, 1000)
softmax = Softmax(memory, "softmax1")
cross_entropy = CrossEntropy(memory, "entropy1")
mean = Mean(memory, "mean")

with memory.hold():
    image, label = data()
    conv1_out = conv1(image)
    relu1_out = relu1(conv1_out)
    pool1_out = pool1(relu1_out, pool_size=3, pool_stride=2, pool_padding=1)
    bottle_out_list = [pool1_out]
    for block in range(len(depth)):
        for i in range(depth[block]):
            stride = 2 if i == 0 and block != 0 else 1
            bottle_out = bottle(bottle_out_list[-1])
            bottle_out_list.append(bottle_out)
    pool2_out = pool2(bottle_out_list[-1], pool_size=7, 
                      pool_type='avg', global_pooling=True)
    fc1_out = fc1(pool2_out)
    softmax_out = softmax(fc1_out)
    cross_entropy_out = cross_entropy(softmax_out)
    mean_out = mean(cross_entropy_out)
    print(str(memory.main_program))
