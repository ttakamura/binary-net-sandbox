import numpy

import binary_batch_normalize_function as bbnf
from chainer import initializers
from chainer import link
from chainer import variable

class BinaryBatchNormalization(link.Link):
    def __init__(self, size, decay=0.9, eps=2e-5, dtype=numpy.float32,
                 use_gamma=True, use_beta=True,
                 initial_gamma=None, initial_beta=None):
        super(BinaryBatchNormalization, self).__init__()
        if use_gamma:
            self.add_param('gamma', size, dtype=dtype)
            if initial_gamma is None:
                initial_gamma = initializers.One()
            initializers.init_weight(self.gamma.data, initial_gamma)
        if use_beta:
            self.add_param('beta', size, dtype=dtype)
            if initial_beta is None:
                initial_beta = initializers.Zero()
            initializers.init_weight(self.beta.data, initial_beta)
        self.add_persistent('avg_mean', numpy.zeros(size, dtype=dtype))
        self.add_persistent('avg_var', numpy.zeros(size, dtype=dtype))
        self.add_persistent('N', 0)
        self.decay = decay
        self.eps = eps

    def __call__(self, x, test=False, finetune=False):
        if hasattr(self, 'gamma'):
            gamma = self.gamma
        else:
            gamma = variable.Variable(self.xp.ones(
                self.avg_mean.shape, dtype=x.dtype), volatile='auto')
        if hasattr(self, 'beta'):
            beta = self.beta
        else:
            beta = variable.Variable(self.xp.zeros(
                self.avg_mean.shape, dtype=x.dtype), volatile='auto')

        if not test:
            if finetune:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay

            func = bbnf.BatchNormalizationFunction(
                self.eps, self.avg_mean, self.avg_var, True, decay)
            ret = func(x, gamma, beta)

            self.avg_mean = func.running_mean
            self.avg_var = func.running_var
        else:
            # Use running average statistics or fine-tuned statistics.
            mean = variable.Variable(self.avg_mean, volatile='auto')
            var = variable.Variable(self.avg_var, volatile='auto')
            ret = bbnf.fixed_batch_normalization(
                x, gamma, beta, mean, var, self.eps)
        return ret

    def start_finetuning(self):
        self.N = 0
