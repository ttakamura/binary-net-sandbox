#!/usr/bin/env python
from __future__ import print_function
import code
import sys
import struct
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from binary_net import BinaryMLP

def pack_vector(vector, d):
  for val in vector:
    d += struct.pack('f',val)
  return d

def write_liner_W(data, filename):
  d = bytearray()
  matrix = np.where(data >= 0, 1, -1).astype(np.float32, copy=False)
  for row in matrix:
    d = pack_vector(row, d)
  with open(filename,'wb') as f:
    f.write(d)

# >>> repl.model.predictor.b1.avg_mean.shape
# (1000,)
# >>> repl.model.predictor.b1.avg_var.shape
# (1000,)
# >>> repl.model.predictor.b1.beta.data.shape
# (1000,)
# >>> repl.model.predictor.b1.gamma.data.shape
# (1000,)
def write_batch_norm(data, filename):
  d = bytearray()
  d = pack_vector(data.avg_mean, d)
  d = pack_vector(data.avg_var, d)
  d = pack_vector(data.beta.data, d)
  d = pack_vector(data.gamma.data, d)
  with open(filename,'wb') as f:
    f.write(d)

def write_input_data(data, filename):
  d = bytearray()
  d = pack_vector(data, d)
  with open(filename,'wb') as f:
    f.write(d)

argvs = sys.argv

unit = 1000
model = L.Classifier(BinaryMLP(784, unit, 10))
chainer.serializers.load_npz(argvs[1], model)

write_liner_W(model.predictor.l1.W.data, 'result/binary_net.l1.W.dat')
write_liner_W(model.predictor.l2.W.data, 'result/binary_net.l2.W.dat')
write_liner_W(model.predictor.l3.W.data, 'result/binary_net.l3.W.dat')

write_batch_norm(model.predictor.b1, 'result/binary_net.b1.dat')
write_batch_norm(model.predictor.b2, 'result/binary_net.b2.dat')
write_batch_norm(model.predictor.b3, 'result/binary_net.b3.dat')

train, test = chainer.datasets.get_mnist()
data, category = train[1206]
write_input_data(data, 'result/binary_net.x.1206.{}.dat'.format(category))

data, category = train[2001]
write_input_data(data, 'result/binary_net.x.2001.{}.dat'.format(category))

data, category = train[3000]
write_input_data(data, 'result/binary_net.x.3000.{}.dat'.format(category))

data, category = train[4000]
write_input_data(data, 'result/binary_net.x.4000.{}.dat'.format(category))

data, category = train[8000]
write_input_data(data, 'result/binary_net.x.8000.{}.dat'.format(category))
