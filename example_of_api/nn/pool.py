from paddle.fluid import Operator
from module import Module
from .util import _single, _pair, _triple

class Pool2d(Module):
    r""" pooling 2d """
    def __init__(self, memory, base_name):
        super(Pool2d, self).__init__()
        self.memory = memory
        self.base_name = base_name
        self.call_count = 0

    def forward(self, 
                input, 
                pool_type='max',
                pool_size=-1, 
                pool_stride=1,
                pool_padding=0,
                use_cudnn=True,
                ceil_mode=False,
                use_mkldnn=False,
                global_pooling=False):
        print input.shape
        if pool_type not in ["max", "avg"]:
            raise ValueError(
                "Unknown pool_type: '%s'. It can only be 'max' or 'avg'.",
                str(pool_type))
        
        if global_pooling is False and pool_size == -1:
            raise ValueError(
                "When the global_pooling is False, pool_size must be passed "
                "and be a valid value. Received pool_size: " + str(pool_size))
        main_block = self.memory.main_program.current_block()    
        out_name = "%s_%d_out" % (self.base_name, self.call_count)
        pool_out = main_block.create_var(name=out_name,
                                         dtype='float32')
        self.memory.add_var(pool_out)
        pool2d_op_desc = main_block.desc.append_op()
        pool2d_op = Operator(block=main_block,
                             desc=pool2d_op_desc,
                             type='pool2d',
                             inputs={'X': input},
                             outputs={'Out': pool_out},
                             attrs={"pooling_type": pool_type,
                                    "ksize": _pair(pool_size),
                                    "global_pooling": global_pooling,
                                    "strides": _pair(pool_stride),
                                    "paddings": _pair(pool_padding),
                                    "use_cudnn": use_cudnn,
                                    "ceil_mode": ceil_mode,
                                    "use_mkldnn": use_mkldnn})
        main_block.ops.append(pool2d_op)
        self.call_count += 1
        return pool_out
