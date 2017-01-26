import numpy
from chainer import function
from chainer.utils import type_check

def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)

class BinaryLinearFunction(function.Function):
    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim >= 2,
            w_type.ndim == 2,
            type_check.prod(x_type.shape[1:]) == w_type.shape[1],
        )
        if n_in.eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def forward(self, inputs):
        x = _as_mat(inputs[0])
        W = inputs[1]
        Wb = numpy.where(W>=0, 1, -1).astype(numpy.float32, copy=False)

        # print("================")
        # hist_x_count, hist_x_guide = numpy.histogram(x)
        # print("X", hist_x_count, hist_x_guide)
        # hist_wb_count, hist_wb_guide = numpy.histogram(Wb)
        # print("Wb", hist_wb_count, hist_wb_guide)

        y = x.dot(Wb.T).astype(x.dtype, copy=False)
        if len(inputs) == 3:
            b = inputs[2]
            y += b
        return y,

    def backward(self, inputs, grad_outputs):
        x = _as_mat(inputs[0])
        W = inputs[1]
        Wb = numpy.where(W>=0, 1, -1).astype(numpy.float32, copy=False)
        gy = grad_outputs[0]

        gx = gy.dot(Wb).astype(x.dtype, copy=False).reshape(inputs[0].shape)
        gW = gy.T.dot(x).astype(W.dtype, copy=False)
        if len(inputs) == 3:
            gb = gy.sum(0)
            return gx, gW, gb
        else:
            return gx, gW

def binary_linear(x, W, b=None):
    if b is None:
        return BinaryLinearFunction()(x, W)
    else:
        return BinaryLinearFunction()(x, W, b)
