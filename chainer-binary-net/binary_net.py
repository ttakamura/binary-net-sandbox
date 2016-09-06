import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

class BinaryMLP(chainer.Chain):
    def __init__(self, n_in, n_units, n_out):
      super(BinaryMLP, self).__init__(
        l1=L.Linear(n_in, n_units),  # first layer
        b1=L.BatchNormalization(n_units),
        l2=L.Linear(n_units, n_units),  # second layer
        b2=L.BatchNormalization(n_units),
        l3=L.Linear(n_units, n_out),  # output layer
        b3=L.BatchNormalization(n_out),
      )
      self.train = True

    def __call__(self, x):
      h1 = F.relu(self.b1(self.l1(x), test=not self.train))
      h2 = F.relu(self.b2(self.l2(h1), test=not self.train))
      return self.b3(self.l3(h2), test=not self.train)
