from paddle.fluid import Operator
from module import Module

class Mean(Module):
    r""" mean of a tensor"""
    def __init__(self, memory, base_name):
        super(Mean, self).__init__()
        self.memory = memory
        self.base_name = base_name
        self.call_count = 0
        
    def forward(self, input, dim=None, keep_dim=False):
        out_name = "%s_%d_out" % (self.base_name, self.call_count)
        start_block = self.memory.startup_program.global_block()
        main_block = self.memory.main_program.current_block()
        out_var = main_block.create_var(name=out_name,
                                        dtype='float32')
        mean_desc = main_block.desc.append_op()
        mean_op = Operator(block=main_block,
                           desc=mean_desc,
                           type='reduce_mean',
                           inputs={'X': input},
                           outputs={'Out': out_var},
                           attrs={'dim': dim if dim != None else [0],
                                  'keep_dim': keep_dim,
                                  'reduce_all': True if dim == None else False})
        self.memory.add_var(out_var)
        main_block.ops.append(mean_op)
        self.call_count += 1
        return out_var
        
