import paddle.fluid as fluid
import contextlib

class Memory(object):
    def __init__(self):
        pass

    def hold(self):
        pass

    def add_weight(self, var):
        pass

    def add_var(self, var):
        pass

class GlobalMemory(Memory):
    def __init__(self):
        self.main_program = fluid.Program()
        self.startup_program = fluid.Program()
        self.weight_param_num = 0
        self.var_param_num = 0

    @contextlib.contextmanager
    def hold(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            yield

    def _cprod_sum(self, var):
        return abs(reduce(lambda x, y: x * y, var.shape, 1))

    def add_weight(self, var):
        self.weight_param_num += self._cprod_sum(var)

    def add_var(self, var):
        self.var_param_num += self._cprod_sum(var)

    
