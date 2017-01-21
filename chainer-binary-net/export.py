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
chainer.serializers.load_npz(argvs[1], model)

d = bytearray()

for v in model.predictor.l1.b.data:
  d += struct.pack('f',v)

with open('result/binary_net.l1.b.dat','wb') as f:
  f.write(d)

code.InteractiveConsole(globals()).interact()
