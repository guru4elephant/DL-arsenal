from paddle.fluid import Operator
from paddle.fluid.initializer import Constant
from module import Module

class ElemAdd(Module):
    r""" elementwise add operation """
    def __init__(self, memory, base_name):
        super(ElemAdd, self).__init__()
        self.memory = memory
        self.base_name = base_name

    def forward(self, input1, input2):
        main_block = self.memory.main_program.current_block()
        out_var = main_block.create_var(name="%s_out" % self.base_name, 
                                        dtype='float32')
        element_add_op_desc = main_block.desc.append_op()
        element_add_op = Operator(block=main_block,
                                  desc=element_add_op_desc,
                                  type='elementwise_add',
                                  inputs={'X': [input1],
                                          'Y': [input2]},
                                  outputs={'Out': [out_var]})
        main_block.ops.append(element_add_op)
        return out_var


class BatchNorm(Module):
    r""" batch normalization """
    def __init__(self, memory, base_name, channel):
        super(BatchNorm, self).__init__()
        self.memory = memory
        self.base_name = base_name
        self.call_count = 0
        self.scale_name = "%s_scale" % base_name
        self.bias_name = "%s_bias" % base_name
        self.mean_name = "%s_mean" % base_name
        self.var_name = "%s_var" % base_name
        start_block = memory.startup_program.global_block()
        main_block = memory.main_program.current_block()
        self.scale = start_block.create_parameter(name=self.scale_name,
                                                  shape=[channel],
                                                  dtype='float32',
                                                  default_initializer=Constant(1.0))
        self.bias = start_block.create_parameter(name=self.bias_name,
                                                 shape=[channel],
                                                 dtype='float32')
        self.mean = start_block.create_parameter(name=self.mean_name,
                                                 initializer=Constant(0.0),
                                                 trainable=False,
                                                 do_model_average=False,
                                                 shape=[channel],
                                                 dtype='float32')
        self.mean.stop_gradient = True
        self.variance = start_block.create_parameter(name=self.var_name,
                                                     initializer=Constant(1.0),
                                                     trainable=False,
                                                     do_model_average=False,
                                                     shape=[channel],
                                                     dtype='float32')
        self.main_scale = main_block.create_parameter(name=self.scale_name,
                                                 shape=[channel],
                                                 dtype='float32')
        self.main_bias = main_block.create_parameter(name=self.bias_name,
                                                shape=[channel],
                                                dtype='float32')
        self.main_mean = main_block.create_parameter(name=self.mean_name,
                                                do_model_average=False,
                                                shape=[channel],
                                                dtype='float32')
        self.main_mean.stop_gradient = True
        self.main_variance = main_block.create_parameter(name=self.var_name,
                                                    do_model_average=False,
                                                    shape=[channel],
                                                    dtype='float32')
        self.main_variance.stop_gradient = True
        self.memory.add_weight(self.variance)
        self.memory.add_weight(self.mean)
        self.memory.add_weight(self.scale)
        self.memory.add_weight(self.bias)
        

    def forward(self, input):
        main_block = self.memory.main_program.current_block()
        batch_norm_op_desc = main_block.desc.append_op()
        batch_norm_out = main_block.create_var(name="%s_bn_out" % self.base_name,
                                               dtype='float32')
        mean_out = self.mean
        variance_out = self.variance
        saved_mean = main_block.create_var(name="%s_mean_out" % self.base_name,
                                           dtype='float32')
        saved_variance = main_block.create_var(name="%s_var_out" % self.base_name,
                                               dtype='float32')
        batch_norm_op = Operator(block=main_block,
                                 desc=batch_norm_op_desc,
                                 type="batch_norm",
                                 inputs={
                                     "X": input,
                                     "Scale": self.main_scale,
                                     "Bias": self.main_bias,
                                     "Mean": self.main_mean,
                                     "Variance": self.main_variance
                                 },
                                 outputs={
                                     "Y": batch_norm_out,
                                     "MeanOut": mean_out,
                                     "VarianceOut": variance_out,
                                     "SavedMean": saved_mean,
                                     "SavedVariance": saved_variance
                                 },
                                 attrs={
                                     "momentum": 0.9,
                                     "epsilon": 1e-5,
                                     "is_test": False,
                                     "use_mkldnn": False,
                                     "fuse_with_relu": False
                                 })
        self.memory.add_var(batch_norm_out)
        self.memory.add_var(mean_out)
        self.memory.add_var(variance_out)
        self.memory.add_var(saved_mean)
        self.memory.add_var(saved_variance)
        main_block.ops.append(batch_norm_op)
        self.call_count += 1
        return batch_norm_out

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
        
