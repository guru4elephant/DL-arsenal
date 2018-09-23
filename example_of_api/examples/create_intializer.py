import paddle.fluid as fluid

main_program = fluid.Program()
cur_block = main_program.current_block()
new_var = cur_block.create_var(name='X', shape=[-1, 23, 48], dtype='float32')
const_init = fluid.initializer.ConstantInitializer()
const_init(new_var, cur_block)
print(str(main_program))
