from paddle.fluid import Operator
from paddle.fluid.initializer import XavierInitializer
from paddle.fluid.initializer import ConstantInitializer
from module import Module

class Linear(Module):
    r"""full connection layer: y = Ax + b
    Args:
    input_dim: size of each input sample
    output_dim: size of each output sample
    bias: If set to False, b is not valid, default True

    Examples:
        >>> fc = Linear(256, 64)
        >>> output = fc(input)
    """
    
    def __init__(self, block, base_name, 
                 input_dim, output_dim, bias=True):
        super(Linear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.base_name = base_name
        self.block = block
        self.w_name = "%s_weight" % base_name
        self.b_name = "%s_bias" % base_name
        self.weight = self.block.create_parameter(name=self.w_name,
                                             dtype='float32',
                                             shape=[input_dim, output_dim],
                                             initializer=XavierInitializer(uniform=True,
                                                                           fan_in=input_dim,
                                                                           fan_out=output_dim))
        if bias:
            self.bias = self.block.create_parameter(name=self.b_name,
                                                    dtype='float32',
                                                    shape=[output_dim],
                                                    initializer=ConstantInitializer(value=0.0))
        self.call_count = 0
        
    def forward(self, input):
        """ create op here"""
        tmp_out_name = "%s_tmp_%d_out" % (self.base_name, 
                                          self.call_count)
        tmp_out_var = self.block.create_var(name=tmp_out_name,
                                            shape=[-1, self.output_dim],
                                            dtype='float32')
        mul_op_desc = self.block.desc.append_op()
        mul_op = Operator(block=self.block, 
                          desc=mul_op_desc, 
                          type='mul',
                          inputs={'X': [input],
                                  'Y': [self.weight]},
                          outputs={'Out': [tmp_out_var]},
                          attrs={'x_num_col_dims': 1,
                                 'y_num_col_dims': 1})
        if self.bias:
            final_out_name = "%s_%d_out" % (self.base_name,
                                            self.call_count)
            final_out_var = self.block.create_var(name=final_out_name,
                                                  shape=[-1, self.output_dim],
                                                  dtype='float32')
            add_op_desc = self.block.desc.append_op()
            add_op = Operator(block=self.block,
                              desc=add_op_desc,
                              type='elementwise_add',
                              inputs={'X': [tmp_out_var],
                                      'Y': [self.bias]},
                              outputs={'Out': [final_out_var]},
                              attrs={'axis': 1})
            return final_out_var
        self.call_count += 1
        return tmp_out_var
