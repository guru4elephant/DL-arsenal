import paddle.fluid as fluid
from .module import Module

class ImageData(Module):
    def __init__(self, block, base_name, dim):
        super(ImageData, self).__init__()
        self.image_name = "%s_image" % base_name
        self.label_name = "%s_label" % base_name
        self.block = block
        self.image = block.create_var(name=self.image_name,
                                      shape=[-1, dim],
                                      dtype='float32',
                                      stop_gradient=True,
                                      lod_level=0,
                                      is_data=True)
        self.label = block.create_var(name=self.label_name,
                                      shape=[-1, 1],
                                      dtype='float32',
                                      stop_gradient=True,
                                      lod_level=0,
                                      is_data=True)

    def forward(self):
        return self.image, self.label
        
