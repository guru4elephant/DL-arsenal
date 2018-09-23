from paddle.fluid import Program, Operator

my_program = Program()
cur_block = my_program.current_block()
abs_input_var = cur_block.create_var(name='abs_input', 
                                     shape=[-1, 32, 32],
                                     dtype='float32')
abs_output_var = cur_block.create_var(name='abs_output',
                                      shape=[-1, 32, 32],
                                      dtype='float32')

op_desc = cur_block.desc.append_op()
op = Operator(block=cur_block, desc=op_desc, type='abs', 
              inputs={'X': [abs_input_var]}, outputs={'Out': [abs_output_var]})

print(str(my_program))
