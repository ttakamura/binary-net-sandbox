#!/usr/bin/env python
from __future__ import print_function
import code
import sys
import struct
import numpy as np
import json

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from binary_net import BinaryMLP
from bst import bst

def bn1(x, data):
  avg_mean = data.avg_mean
  avg_var  = data.avg_var
  beta     = data.beta.data
  gamma    = data.gamma.data
  x_hat = (x - avg_mean) / np.sqrt(avg_var + 0.0001)
  y = (gamma * x_hat) + beta
  return y

def bn2(x, data):
  avg_mean = data.avg_mean
  avg_var  = data.avg_var
  beta     = data.beta.data
  gamma    = data.gamma.data
  t = avg_mean - ((beta * np.sqrt(avg_var + 0.0001)) / gamma)
  y = x - t
  return y

def forward_linear(layer, x, path):
  y = layer(x)
  with open(path, "w") as f:
    f.writelines([str(int(val))+"\n" for val in y.data[0, :].tolist()])
  return y

def forward_bn(layer, y, path):
  z = layer(y, test=True)
  with open(path, "w") as f:
    f.writelines([str(val)+"\n" for val in z.data[0, :].tolist()])
  # with open("tmp/output_bn_normal.txt", "w") as f:
  #   bn1_z = bn1(y.data[0,:], model.predictor.b1)
  #   f.writelines([str(val)+"\n" for val in bn1_z.tolist()])
  # with open("tmp/output_bn_thresh.txt", "w") as f:
  #   bn2_z = bn2(y.data[0,:], model.predictor.b1)
  #   f.writelines([str(val)+"\n" for val in bn2_z.tolist()])
  return z

argvs = sys.argv
unit  = 1000
model = L.Classifier(BinaryMLP(784, unit, 10))

if len(argvs) > 1:
  chainer.serializers.load_npz(argvs[1], model)

  x  = np.ones((1, 784), dtype=np.float32) * 128
  y1 = forward_linear(model.predictor.l1, x, "tmp/output_y.txt")
  z1 = forward_bn(model.predictor.b1, y1, "tmp/output_bn.txt")
  h1 = bst(z1)

  y2 = forward_linear(model.predictor.l2, h1, "tmp/output_y2.txt")
  z2 = forward_bn(model.predictor.b2, y2, "tmp/output_bn2.txt")
  h2 = bst(z2)

  y3 = forward_linear(model.predictor.l3, h2, "tmp/output_y3.txt")
  z3 = forward_bn(model.predictor.b3, y3, "tmp/output_bn3.txt")

#
# chainer.serializers.load_npz(argvs[1], model)
# train, test = chainer.datasets.get_mnist()
# row = train[0]
# data, teacher = row
# model.predictor(data.reshape(1, 784))
#

# code.InteractiveConsole(globals()).interact()
