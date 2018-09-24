import paddle.fluid as fluid
from .module import Module

class ImageData(Module):
    def __init__(self, memory, base_name, dim):
        super(ImageData, self).__init__()
        self.image_name = "%s_image" % base_name
        self.label_name = "%s_label" % base_name
        self.memory = memory
        start_block = memory.startup_program.global_block()
        main_block = memory.main_program.current_block()
        self.image = main_block.create_var(name=self.image_name,
                                           shape=[-1, dim],
                                           dtype='float32',
                                           stop_gradient=True,
                                           lod_level=0,
                                           is_data=True)
        self.memory.add_var(self.image)
        self.label = main_block.create_var(name=self.label_name,
                                           shape=[-1, 1],
                                           dtype='int64',
                                           stop_gradient=True,
                                           lod_level=0,
                                           is_data=True)
        self.memory.add_var(self.label)

    def forward(self):
        return self.image, self.label
        
