import paddle.fluid as fluid
from paddle.fluid import Program
from paddle.fluid.op import Operator

cur_program = fluid.Program()
cur_block = cur_program.current_block()
new_variable = cur_block.create_var(name="X",
                                    shape=[-1, 1],
                                    dtype='int64')

counter = cur_program.global_block().create_var(persistable=True,
                                                name="counter",
                                                dtype='int64',
                                                shape=[-1, 1])

cur_block.append_op(type='elementwise_add',
                    inputs={'X':[new_variable], 'Y':[counter]}, outputs={'Out':[counter]})

print(str(cur_program))
'''
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
'''
