
''' a pytorch like interface for neural network layer '''
class Module(object):
    r""" Base class for all layers.
    example: To be added here
    """
    def __init__(self):
        self.training = True

    def forward(self, *input):
        r""" forward computation for each call
        Should be overridden by all subclasses.
        """
        raise NotImplementedError
    
    def __call__(self, *input, **kwargs):
        print "calling ", self.__class__.__name__
        result = self.forward(*input, **kwargs)
        return result
