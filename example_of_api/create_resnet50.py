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
from nn.util import volumn, find_var_by_name
import numpy as np
from paddle.fluid import debugger

batch_size = 128

def print_var_by_id(id):
    print "  vars {"
    print "    name: %d_rep_var" % id
    print "    type {"
    print "      type: LOD_TENSOR"
    print "      lod_tensor {"
    print "        tensor {"
    print "          data_type: FP32"
    print "          dims: 256"
    print "        }"
    print "      }"
    print "    }"
    print "  }"    

def print_op(op):
    print "  ops {"
    lines = str(op).split("\n")
    for line in lines:
        print "    " + line
    print "  }"

def print_var(var):
    print "  vars {"
    lines = str(var).split("\n")
    for line in lines:
        print "    " + line
    print "  }"

renamed_var_dict = {}

def rewrite_ops(ops, pool_vars, checkpoints, weight_dict):
    pool_idx = 0
    local_dict = {}
    for op in ops:
        for i, name in enumerate(op.input_arg_names):
            if name not in checkpoints and name not in weight_dict:
                if name in local_dict:
                    op.rename_input(op.input_arg_names[i], local_dict[name])
                else:
                    renamed_var_dict[name] = 1
                    op.rename_input(op.input_arg_names[i], pool_vars[pool_idx])
                    local_dict[name] = pool_vars[pool_idx]
                    pool_idx += 1
        for i, name in enumerate(op.output_arg_names):
            if name not in checkpoints and name not in weight_dict:
                if name in local_dict:
                    op.rename_output(op.output_arg_names[i], local_dict[name])
                else:
                    renamed_var_dict[name] = 1
                    op.rename_output(op.output_arg_names[i], pool_vars[pool_idx])
                    local_dict[name] = pool_vars[pool_idx]
                    pool_idx += 1
    return pool_idx

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
            self.conv3 = ConvBnNet(memory, "%s_conv3" % basename, in_channel, out_channel * 4, 1, stride)
        self.final_out_channel = out_channel * 4
        self.add = ElemAdd(memory, "%s_shortcut" % basename)


    def forward(self, input):
        conv0 = self.conv0(input)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        if self.stride != 1 or self.in_channel != self.out_channel * 4:
            conv3 = self.conv3(input)
            result = self.add(conv3, conv2)
            return result
        return self.add(input, conv2)


class ConvBnNet(Module):
    def __init__(self, memory, basename, in_channel, out_channel, 
                 kernel_size, stride, padding=1, dilation=1,
                 groups=1, act='relu'):
        super(ConvBnNet, self).__init__()
        self.memory = memory
        padding = (kernel_size - 1) // 2
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

forward_memory = GlobalMemory()
backward_memory = GlobalMemory()

data = ImageData(memory, "data", [3, 224, 224])

depth = [3, 4, 6, 3]
channels = [64, 64, 128, 256, 512]
conv1 = ConvBnNet(memory, "first_conv_bn", 3, channels[0], 
                  7, 2)
relu1 = Relu(memory, "first_relu")
pool1 = Pool2d(memory, "first_pool")

last = 64
bottleneck_list = []
for block in range(len(depth)):
    for i in range(depth[block]):
        stride = 2 if i == 0 and block != 0 else 1
        bottle = BottleneckBlock(memory, "bottleneck_%d_%d" % (block, i), 
                                 last, channels[block+1], stride=stride)
        last = bottle.final_out_channel
        bottleneck_list.append(bottle)

pool2 = Pool2d(memory, "second_pool")
fc1 = Linear(memory, "fc1", 2048, 1000)
softmax = Softmax(memory, "softmax1")
cross_entropy = CrossEntropy(memory, "entropy1")
mean = Mean(memory, "mean")

partitions = []

pool_vars = [str(i) + "_rep_var" for i in range(100)]

weight_dict = {}

with memory.hold():
    image, label = data()
    conv1_out = conv1(image)
    relu1_out = relu1(conv1_out)
    pool1_out = pool1(relu1_out, pool_size=3, pool_stride=2, pool_padding=1)
    checkpoint_vars = [pool1_out.name, "%s@GRAD" % pool1_out.name]
    bottle_out_list = [pool1_out]
    bottle_idx = 0
    checkpoint_idx = 3
    for block in range(len(depth)):
        for i in range(depth[block]):
            stride = 2 if i == 0 and block != 0 else 1
            bottle_out = bottleneck_list[bottle_idx](bottle_out_list[-1])
            checkpoint_idx += 1
            bottle_out_list.append(bottle_out)
            bottle_idx += 1
            checkpoint_vars.append(bottle_out.name)
            checkpoint_vars.append("%s@GRAD" % bottle_out.name)
    pool2_out = pool2(bottle_out_list[-1], pool_size=7, 
                      pool_type='avg', global_pooling=True)
    checkpoint_idx += 1
    fc1_out = fc1(pool2_out)
    softmax_out = softmax(fc1_out)
    cross_entropy_out = cross_entropy(softmax_out, label)
    mean_out = mean(cross_entropy_out)
    checkpoint_vars.append(mean_out.name)
    checkpoint_vars.append("%s@GRAD" % mean_out.name)
    main_block = memory.main_program.current_block()
    checkpoints_width = 0
    sgd_opt = fluid.optimizer.SGDOptimizer(learning_rate=0.1)
    opt_ops, weight_and_grad = sgd_opt.minimize(mean_out, startup_program=memory.startup_program)
    for x in weight_and_grad:
        weight_dict[x[0].name] = 1
    print(str(memory.main_program))
    sys.exit(-1)


opt_type = {}
for op in memory.main_program.current_block().ops:
    if op.type in opt_type:
        opt_type[op.type] = 1

forward_memory.startup_program = memory.startup_program.clone()
forward_memory.main_program = memory.main_program.clone()
backward_memory.startup_program = memory.startup_program.clone()
backward_memory.main_program = memory.main_program.clone()
max_var_num = 0

final_forward_partitions = []
with forward_memory.hold():
    fp = []
    bp = []
    forward_partitions = []
    backward_partitions = []
    for i, op in enumerate(forward_memory.main_program.current_block().ops):
        if op.type in opt_type:
            continue
        is_backward = False
        for name in op.output_arg_names:
            if "@GRAD" in name:
                is_backward = True
        for name in op.input_arg_names:
            if "@GRAD" in name:
                is_backward = True
        if "_grad" in op.type:
            is_backward = True
        is_partition = False
        for name in op.output_arg_names:
            if name in checkpoint_vars:
                is_partition = True
                break
        if is_backward:
            continue
        else:
            fp.append(op)
            if is_partition:
                final_forward_partitions.append(fp)
                fp = []


with backward_memory.hold():
    forward_partitions = []
    backward_partitions = []
    fp = []
    bp = []
    
    for i, op in enumerate(backward_memory.main_program.current_block().ops):
        if op.type in opt_type:
            sgd_ops.append(op)
            continue
        is_backward = False
        for name in op.output_arg_names:
            if "@GRAD" in name:
                is_backward = True
        if "_grad" in op.type:
            is_backward = True
        is_partition = False
        for name in op.output_arg_names:
            if name in checkpoint_vars:
                is_partition = True
                break
        if is_backward:
            continue
        else:
            fp.append(op)
            if is_partition:
                forward_partitions.append(fp)
                fp = []
                
    ff_partition_idx = len(forward_partitions) - 1
    fp = []
    bp = []
    
    for op in backward_memory.main_program.current_block().ops:
        if op in opt_ops:
            continue
        is_backward = False
        for name in op.output_arg_names:
            if "@GRAD" in name:
                is_backward = True
        if "_grad" in op.type:
            is_backward = True
        is_partition = False
        for name in op.output_arg_names:
            if name in checkpoint_vars:
                is_partition = True
                break
        if is_backward:
            bp.append(op)
            if is_partition:
                backward_partitions.append(bp)
                ff_partition_idx -= 1
                bp = forward_partitions[ff_partition_idx]
                
    for part in backward_partitions:
        var_num = rewrite_ops(part, pool_vars, checkpoint_vars, weight_dict)
        if var_num > max_var_num:
            max_var_num = var_num

print "blocks {"
print "  idx: 0"
print "  parent_idx: -1"
with memory.hold():
    name_keys = memory.main_program.current_block().vars.keys()
    vars = memory.main_program.current_block().vars
    for name in name_keys:
        if name not in renamed_var_dict:
            print_var(vars[name])
    for i in range(max_var_num):
        print_var_by_id(i)
        
with forward_memory.hold():
    for part in final_forward_partitions:
        var_num = rewrite_ops(part, pool_vars, checkpoint_vars, weight_dict)
        if var_num > max_var_num:
            max_var_num = var_num
        for op in part:
            print_op(op)

with backward_memory.hold():
    for part in backward_partitions:
        var_num = rewrite_ops(part, pool_vars, checkpoint_vars, weight_dict)
        if var_num > max_var_num:
            max_var_num = var_num
        for op in part:
            print_op(op)

for op in opt_ops:
    print_op(op)

print "}"
