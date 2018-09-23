import paddle.fluid as fluid

my_program = Program()
cur_block = my_program.current_block()
data_var = cur_block.create_var(name='image',
                                shape=[-1, 224, 224],
                                dtype='float32',
                                stop_gradient=True,
                                lod_level=0,
                                is_data=True)

