from paddle.fluid import Operator
from module import Module

class Relu(Module):
    def __init__(self, memory, base_name):
        self.memory = memory
        self.base_name = base_name
        self.call_count = 0

    def forward(self, input):
        main_block = self.memory.main_program.current_block()
        out_name = "%s_%d_out" % (self.base_name, self.call_count)
        out_var = main_block.create_var(name=out_name,
                                        dtype='float32')
        self.memory.add_var(out_var)
        relu_op_desc = main_block.desc.append_op()
        relu_op = Operator(block=main_block,
                           desc=relu_op_desc,
                           type='relu',
                           inputs={'X': input},
                           outputs={'Out':out_var})
        main_block.ops.append(relu_op)
        self.call_count += 1
        return out_var

