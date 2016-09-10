import numpy
from chainer import cuda
from chainer import function
from chainer.utils import type_check

def _as4darray(arr):
    if arr.ndim == 0:
        return arr.reshape(1, 1, 1, 1)
    elif arr.ndim == 4:
        return arr
    else:
        return arr.reshape(arr.shape[0], -1, 1, 1)

def _xhat(x, mean, std, expander):
    x_mu = x - mean[expander]
    x_mu /= std[expander]
    return x_mu

class BatchNormalizationFunction(function.Function):
    def __init__(self, eps=2e-5, mean=None, var=None, train=False,
                 decay=0.9, use_cudnn=True):
        self.running_mean = mean
        self.running_var = var
        self.train = train
        self.eps = eps
        self.use_cudnn = use_cudnn
        self.mean_cache = None
        self.decay = decay

    def check_type_forward(self, in_types):
        n_in = in_types.size().eval()
        if n_in != 3 and n_in != 5:
            raise type_check.InvalidType(
                '%s or %s' % (in_types.size() == 3, in_types.size() == 5),
                '%s == %s' % (in_types.size(), n_in))
        x_type, gamma_type, beta_type = in_types[:3]
        M = gamma_type.ndim.eval()
        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim >= gamma_type.ndim + 1,
            x_type.shape[1:1 + M] == gamma_type.shape,
            # TODO(beam2d): Check shape
            gamma_type.dtype == x_type.dtype,
            beta_type.dtype == x_type.dtype,
            gamma_type.shape == beta_type.shape,
        )
        if len(in_types) == 5:
            mean_type, var_type = in_types[3:]
            type_check.expect(
                mean_type.dtype == x_type.dtype,
                mean_type.shape == gamma_type.shape,
                var_type.dtype == x_type.dtype,
                var_type.shape == gamma_type.shape,
            )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, gamma, beta = inputs[:3]
        if self.train:
            if self.running_mean is None:
                self.running_mean = xp.zeros_like(gamma)
                self.running_var = xp.zeros_like(gamma)
            else:
                self.running_mean = xp.array(self.running_mean)
                self.running_var = xp.array(self.running_var)
        elif len(inputs) == 5:
            self.fixed_mean = inputs[3]
            self.fixed_var = inputs[4]

        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
        gamma = gamma[expander]
        beta = beta[expander]

        if self.train:
            axis = (0,) + tuple(range(head_ndim, x.ndim))
            mean = x.mean(axis=axis)
            var = x.var(axis=axis)
            var += self.eps
        else:
            mean = self.fixed_mean
            var = self.fixed_var
        self.std = xp.sqrt(var, dtype=var.dtype)

        self.x_hat = _xhat(x, mean, self.std, expander)
        y = gamma * self.x_hat
        y += beta

        m = x.size // gamma.size
        adjust = m / max(m - 1., 1.)  # unbiased estimation
        self.running_mean *= self.decay
        temp_ar = xp.array(mean)
        temp_ar *= (1 - self.decay)
        self.running_mean += temp_ar
        del temp_ar
        self.running_var *= self.decay
        temp_ar = xp.array(var)
        temp_ar *= (1 - self.decay) * adjust
        self.running_var += temp_ar
        del temp_ar

        return y,

    def backward(self, inputs, grad_outputs):
        x, gamma = inputs[:2]
        gy = grad_outputs[0]
        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
        m = gamma.dtype.type(x.size // gamma.size)
        axis = (0,) + tuple(range(head_ndim, x.ndim))
        xp = cuda.get_array_module(x)
        if len(inputs) == 5:
            # This case is unlikely to be used in practice and so does not
            # need to be optimized for performance.
            mean = inputs[3]
            var = inputs[4]
            std = xp.sqrt(var, dtype=var.dtype)
            gs = gamma / std
            gbeta = gy.sum(axis=axis)
            x_hat = _xhat(x, mean, std, expander)
            ggamma = (gy * x_hat).sum(axis=axis)
            gmean = -gs * gbeta
            gvar = -0.5 * gamma / var * ggamma
            gx = gs[expander] * gy
            return gx, ggamma, gbeta, gmean, gvar

        # Note: If length of inputs is not 5, we must be in train mode.
        assert self.train
        if xp is not numpy and cuda.cudnn_enabled and self.use_cudnn and \
                self.cudnn_dim_ok and _cudnn_version >= 5000:
            # Note: cuDNN batch normalization backward only works in
            # "training mode." That is, it does not support
            # computing gradients in fixed-mean-variance mode, because there
            # is normally no reason to call backward()
            # while in test/evaluation mode.
            dtype = x.dtype
            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(_as4darray(x))
            derivedBnDesc = cudnn.create_uninitialized_tensor_descriptor()
            libcudnn.deriveBNTensorDescriptor(derivedBnDesc.value,
                                              x_desc.value, self.mode)
            one = numpy.array(1, dtype=dtype).ctypes
            zero = numpy.array(0, dtype=dtype).ctypes
            gx = cuda.cupy.empty_like(x)
            ggamma = cuda.cupy.empty_like(gamma)
            gbeta = cuda.cupy.empty_like(gamma)
            libcudnn.batchNormalizationBackward(
                handle, self.mode, one.data, zero.data,
                one.data, zero.data, x_desc.value, x.data.ptr,
                x_desc.value, gy.data.ptr, x_desc.value, gx.data.ptr,
                derivedBnDesc.value, gamma.data.ptr,
                ggamma.data.ptr, gbeta.data.ptr,
                self.eps, self.mean_cache.data.ptr, self.var_cache.data.ptr)
        else:
            gbeta = gy.sum(axis=axis)
            ggamma = (gy * self.x_hat).sum(axis=axis)
            if xp is numpy:
                gx = (gamma / self.std)[expander] * (
                    gy - (self.x_hat * ggamma[expander] + gbeta[expander]) / m)
            else:
                inv_m = numpy.float32(1) / m
                gx = cuda.elementwise(
                    'T gy, T x_hat, T gamma, T std, T ggamma, T gbeta, \
                    T inv_m',
                    'T gx',
                    'gx = (gamma / std) * (gy - (x_hat * ggamma + gbeta) * \
                    inv_m)',
                    'bn_bwd')(gy, self.x_hat, gamma[expander],
                              self.std[expander], ggamma[expander],
                              gbeta[expander], inv_m)
        return gx, ggamma, gbeta


def batch_normalization(x, gamma, beta, eps=2e-5, running_mean=None,
                        running_var=None, decay=0.9, use_cudnn=True):
    return BatchNormalizationFunction(eps, running_mean, running_var, True,
                                      decay, use_cudnn)(x, gamma, beta)


def fixed_batch_normalization(x, gamma, beta, mean, var, eps=2e-5,
                              use_cudnn=True):
    return BatchNormalizationFunction(eps, None, None, False, 0.0,
                                      use_cudnn)(x, gamma, beta, mean, var)
