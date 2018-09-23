import paddle.fluid as fluid
from paddle.fluid import Operator

my_program = fluid.Program()
cur_block = my_program.current_block()
lr = 0.1
abs_input_var = cur_block.create_parameter(name='abs_input', 
                                           shape=[1],
                                           dtype='float32',
                                           initializer=fluid.initializer.ConstantInitializer())
abs_output_var = cur_block.create_var(name='loss',
                                      shape=[1],
                                      dtype='float32')

op_desc = cur_block.desc.append_op()
op = Operator(block=cur_block, desc=op_desc, type='abs', 
              inputs={'X': [abs_input_var]}, outputs={'Out': [abs_output_var]})

sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=lr)
sgd_optimizer.minimize(abs_output_var, startup_program=my_program)

print(str(my_program))
