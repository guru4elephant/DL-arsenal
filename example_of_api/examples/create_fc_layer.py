from paddle.fluid import Program
from paddle.fluid import Operator
from paddle.fluid.initializer import XavierInitializer
from paddle.fluid.initializer import ConstantInitializer
use_mkldnn = False
my_program = Program()
cur_block = my_program.current_block()
# implement y = Wx + b layer, input variable: W, x, b, output variable: y
# initlizer W->Xavier initlization, b->Constant initialization
x_var = cur_block.create_var(name='fc_x',
                             shape=[-1, 128],
                             dtype='float32')
y_var = cur_block.create_var(name='fc_y',
                             shape=[-1, 64],
                             dtype='float32')
Wx_var = cur_block.create_var(name='fc_Wx',
                              shape=[-1, 64],
                              dtype='float32')
xavier_init = XavierInitializer(uniform=True, fan_in=128, fan_out=64)
const_init = ConstantInitializer(value=0.0)
W_var = cur_block.create_parameter(name='fc_W',
                                   dtype='float32',
                                   shape=[128, 64],
                                   initializer=xavier_init)
b_var = cur_block.create_parameter(name='fc_b', 
                                   dtype='float32',
                                   shape=[64],
                                   initializer=const_init)
mul_op_desc = cur_block.desc.append_op()
mul_op = Operator(block=cur_block, desc=mul_op_desc, type='mul',
                  inputs={'X':x_var, 'Y':W_var}, outputs={'Out': Wx_var},
                  attrs={"x_num_col_dims": 1,
                         "y_num_col_dims": 1})

add_op_desc = cur_block.desc.append_op()
add_op = Operator(block=cur_block, desc=add_op_desc, type='elementwise_add', 
                  inputs={'X': [Wx_var], 'Y':[b_var]}, outputs={'Out': [y_var]},
                  attrs={'axis': 1})
print(my_program)
