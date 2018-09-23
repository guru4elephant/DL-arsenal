import paddle.fluid as fluid

cur_program = fluid.Program()
cur_block = cur_program.current_block()
new_parameter = cur_block.create_parameter(dtype='float32', 
                                           shape=[32, 48],
                                           initializer=fluid.initializer.ConstantInitializer())
print(str(cur_program))

