from paddle.fluid import Operator
from paddle.fluid.initializer import Normal
from paddle.fluid.initializer import ConstantInitializer
from module import Module
from .util import _single, _pair, _triple, _quadruple, volumn

class Conv2d(Module):
    r""" convolution 2d """
    def __init__(self, memory, base_name,
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__()
        self.memory = memory
        self.base_name = base_name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        start_block = memory.startup_program.global_block()
        main_block = memory.main_program.current_block()
        conv_name = "%s_w" % self.base_name
        self.kernel_size = _pair(self.kernel_size)
        self.stride = _pair(self.stride)
        self.padding = _pair(self.padding)
        self.dilation = _pair(self.dilation)
        def _get_default_param_initializer():
            std = (2.0 / (self.kernel_size[0]**2 * self.in_channels))**0.5
            return Normal(0.0, std, 0)
        weight_shape = [self.out_channels, self.in_channels] + self.kernel_size
        self.conv_weight = start_block.create_parameter(name=conv_name,
                                                        dtype='float32',
                                                        shape=weight_shape,
                                                        with_initlializer=True,
                                                        initializer=_get_default_param_initializer())
        
        self.memory.add_weight(self.conv_weight)
        self.main_conv_weight = main_block.create_parameter(name=conv_name,
                                                            dtype='float32',
                                                            shape=weight_shape)
        if bias:
            self.b_name = "%s_bias" % self.base_name
            self.bias = start_block.create_parameter(name=self.b_name,
                                                     dtype='float32',
                                                     shape=[self.out_channels],
                                                     with_initializer=True,
                                                     intializer=ConstantInitializer(value=0.0))
            self.memory.add_weight(self.bias)
            self.main_bias = main_block.create_parameter(name=self.b_name,
                                                         dtype='float32',
                                                         shape=[self.out_channels])
        self.call_count = 0

    def __repr__(self):
        main_str = self.__class__.__name__
        return main_str

    def forward(self, 
                input, 
                use_cudnn=True,
                use_mkldnn=True):
        main_block = self.memory.main_program.current_block()
        conv_op_desc = main_block.desc.append_op()
        conv_out = main_block.create_var(name="%s_out" % self.base_name,
                                         dtype='float32')
        conv_op = Operator(block=main_block,
                           desc=conv_op_desc,
                           type='conv2d',
                           inputs={'Input': input,
                                   'Filter': self.main_conv_weight},
                           outputs={'Output': conv_out},
                           attrs={'strides': self.stride,
                                  'paddings': self.padding,
                                  'dilations': self.dilation,
                                  'groups': self.groups,
                                  'use_cudnn': use_cudnn,
                                  'use_mkldnn': use_mkldnn})
        main_block.ops.append(conv_op)
        if self.bias:
            # add bias
            final_out_name = "%s_%d_out" % (self.base_name, 
                                            self.call_count)
            final_out_var = main_block.create_var(name=final_out_name,
                                                  shape=conv_out.shape,
                                                  dtype='float32')
            add_op_desc = main_block.desc.append_op()
            add_op = Operator(block=main_block,
                              desc=add_op_desc,
                              type='elementwise_add',
                              inputs={'X': [conv_out],
                                      'Y': [self.bias]},
                              outputs={'Out': [final_out_var]},
                              attrs={'axis': 1})
            main_block.ops.append(add_op)
            self.memory.add_var(final_out_var)
            return final_out_var
        self.call_count += 1
        return conv_out



