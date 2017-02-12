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

argvs = sys.argv

unit = 1000
model = L.Classifier(BinaryMLP(784, unit, 10))

if len(argvs) > 1:
  chainer.serializers.load_npz(argvs[1], model)

  x = np.ones((1, 784), dtype=np.float32) * 128
  y = model.predictor.l1(x)
  print("y[100] {}".format(y.data[0, 500]))
  print("y[200] {}".format(y.data[0, 500]))
  print("y[300] {}".format(y.data[0, 500]))
  with open("tmp/output_y.txt", "w") as f:
    f.writelines([str(int(val))+"\n" for val in y.data[0, :].tolist()])

  z = model.predictor.b1(y, test=True)
  print("z[100] {}".format( z.data[0, 100]))
  print("z[200] {}".format( z.data[0, 200]))
  print("z[300] {}".format( z.data[0, 300]))
  with open("tmp/output_bn.txt", "w") as f:
    f.writelines([str(val)+"\n" for val in z.data[0, :].tolist()])

  with open("tmp/output_bn_normal.txt", "w") as f:
    bn1_z = bn1(y.data[0,:], model.predictor.b1)
    f.writelines([str(val)+"\n" for val in bn1_z.tolist()])

  with open("tmp/output_bn_thresh.txt", "w") as f:
    bn2_z = bn2(y.data[0,:], model.predictor.b1)
    f.writelines([str(val)+"\n" for val in bn2_z.tolist()])

# code.InteractiveConsole(globals()).interact()
