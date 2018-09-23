import paddle.fluid as fluid

startup_program = fluid.Program()
main_program = fluid.Program()
with fluid.program_guard(main_program=main_program, startup_program=startup_program):
    cur_block = main_program.current_block()
    new_var = cur_block.create_var(name="X", shape=[-1, 23, 48], dtype='float32')
    new_var2 = cur_block.create_var(name='Y', shape=[-1, 1], dtype='int32')
    print(str(main_program))
    for var in main_program.list_vars():
        print(var)
