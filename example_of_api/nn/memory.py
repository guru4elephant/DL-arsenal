import paddle.fluid as fluid
import contextlib

class Memory(object):
    def __init__(self):
        pass

    def hold(self):
        pass

class GlobalMemory(Memory):
    def __init__(self):
        self.main_program = fluid.Program()
        self.startup_program = fluid.Program()

    @contextlib.contextmanager
    def hold(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            yield
