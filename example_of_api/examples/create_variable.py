import paddle.fluid as fluid

cur_program = fluid.Program()
cur_block = cur_program.current_block()
new_variable = cur_block.create_var(name="X",
                                    shape=[-1, 23, 48],
                                    dtype='float32')

print("variable shape")
print(new_variable.shape)
print("variable name")
print(new_variable.name)
print("dtype")
print(new_variable.dtype)
print("lod level")
print(new_variable.lod_level)
print("is persistable")
print(new_variable.persistable)

