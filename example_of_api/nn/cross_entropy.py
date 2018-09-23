from paddle.fluid import Operator
from module import Module

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
        main_block.ops.append(loss_op)
        return loss_var
