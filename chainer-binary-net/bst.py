import numpy
from chainer import function
from chainer.utils import type_check

class BST(function.Function):
    def __init__(self):
        pass

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types
        type_check.expect(
            x_type.dtype == numpy.float32,
        )

    def forward_cpu(self, x):
        y = x[0]
        y = numpy.where(y>=0, 1, -1).astype(numpy.float32, copy=False)
        return y,

    def backward_cpu(self, x, gy):
        gx = gy[0].copy()
        zero_indices = numpy.abs(x[0]) > 1
        gx[zero_indices] = 0
        return gx,

def bst(x):
    return BST()(x)
