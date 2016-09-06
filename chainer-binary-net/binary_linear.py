import math

from binary_linear_function import binary_linear
from chainer import initializers
from chainer import link

class BinaryLinear(link.Link):
    def __init__(self, in_size, out_size, wscale=1, bias=0, nobias=False,
                 initialW=None, initial_bias=None):
        super(BinaryLinear, self).__init__()
        self.initialW = initialW
        self.wscale = wscale
        self.out_size = out_size

        if in_size is None:
            self.add_uninitialized_param('W')
        else:
            self._initialize_params(in_size)

        if nobias:
            self.b = None
        else:
            self.add_param('b', out_size)
            if initial_bias is None:
                initial_bias = bias
            initializers.init_weight(self.b.data, initial_bias)

    def _initialize_params(self, in_size):
        self.add_param('W', (self.out_size, in_size))
        # For backward compatibility, the scale of weights is proportional to
        # the square root of wscale.
        initializers.init_weight(self.W.data, self.initialW,
                                 scale=math.sqrt(self.wscale))

    def __call__(self, x):
        if self.has_uninitialized_params:
            self._initialize_params(x.shape[1])
        return binary_linear(x, self.W, self.b)
