from paddle.fluid import Operator
from module import Module

class Softmax(Module):
    r""" softmax over a tensor"""
    def __init__(self, memory, base_name):
        super(Softmax, self).__init__()
        self.memory = memory
        self.base_name = base_name
        self.call_count = 0
    
    def forward(self, input, use_cudnn=False):
        out_name = "%s_%d_out" % (self.base_name, self.call_count)
        start_block = self.memory.startup_program.global_block()
        main_block = self.memory.main_program.current_block()
        out_var = main_block.create_var(name=out_name,
                                         dtype='float32')
        softmax_desc = main_block.desc.append_op()
        softmax_op = Operator(block=main_block,
                              desc=softmax_desc,
                              type='softmax',
                              inputs={'X':[input]},
                              outputs={'Out':[out_var]},
                              attrs={"use_cudnn":use_cudnn})
        self.memory.add_var(out_var)
        main_block.ops.append(softmax_op)
        self.call_count += 1
        return out_var

class CrossEntropy(Module):
    r""" cross entropy loss between label distribution and 
    probability distribution"""
    def __init__(self, memory, base_name):
        super(CrossEntropy, self).__init__()
        self.memory = memory
        self.base_name = base_name
    
    def forward(self, softmax, label):
        loss_name = "%s_loss" % self.base_name
        start_block = self.memory.startup_program.global_block()
        main_block = self.memory.main_program.current_block()
        loss_var = main_block.create_var(name=loss_name,
                                         dtype='float32')
        loss_desc = main_block.desc.append_op()
        loss_op = Operator(block=main_block,
                           desc=loss_desc,
                           type='cross_entropy',
                           inputs={'X':[softmax],
                                   'Label':[label]},
                           outputs={'Y':[loss_var]},
                           attrs={"soft_label":False})
        self.memory.add_var(loss_var)
        main_block.ops.append(loss_op)
        return loss_var

class SoftmaxWithCrossEntropy(Module):
    r""" cross entropy with softmax"""
    def __init__(self, memory, base_name):
        super(SoftmaxWithCrossEntropy, self).__init__()
        self.memory = memory
        self.base_name = base_name

    def forward(self, logits, label):
        softmax_out_name = "%s_softmax" % self.base_name
        loss_name = "%s_loss" % self.base_name
        start_block = self.memory.startup_program.global_block()
        main_block = self.memory.main_program.current_block()
        softmax_out_var = main_block.create_var(name=softmax_out_name,
                                                dtype='float32')
        loss_var = main_block.create_var(name=loss_name,
                                         dtype='float32')
        loss_desc = main_block.desc.append_op()
        loss_op = Operator(block=main_block,
                           desc=loss_desc,
                           type='softmax_with_cross_entropy',
                           inputs={'Logits': logits,
                                   'Label': label},
                           outputs={'Softmax': softmax_out_var,
                                    'Loss': loss_var},
                           attrs={'soft_label': False})
        self.memory.add_var(loss_var)
        self.memory.add_var(softmax_out_var)
        main_block.ops.append(loss_op)
        return loss_var
