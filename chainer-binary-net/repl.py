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

argvs = sys.argv

unit = 1000
model = L.Classifier(BinaryMLP(784, unit, 10))

if len(argvs) > 1:
  chainer.serializers.load_npz(argvs[1], model)

  x = np.ones((1, 784), dtype=np.float32) * 128
  y = model.predictor.l1(x)
  print("y[0] {}".format(y.data[0, 0]))
  print("y[500] {}".format(y.data[0, 500]))
  print("y[999] {}".format(y.data[0, 999]))

  z = model.predictor.b1(y)
  zb = np.where(z.data >= 0, 1, -1).astype(np.float32, copy=False)
  print("zb[0] {}".format( zb.data[0, 0] ))
  print("zb[100] {}".format( zb.data[0, 100]))
  print("zb[200] {}".format( zb.data[0, 200]))
  print("zb[300] {}".format( zb.data[0, 300]))
  print("zb[400] {}".format( zb.data[0, 400]))
  print("zb[500] {}".format( zb.data[0, 500]))
  print("zb[600] {}".format( zb.data[0, 600]))
  print("zb[700] {}".format( zb.data[0, 700]))
  print("zb[999] {}".format( zb.data[0, 999]))

# code.InteractiveConsole(globals()).interact()
