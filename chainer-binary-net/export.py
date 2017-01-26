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

def write_liner_W(data, filename):
  d = bytearray()
  data = np.where(data >= 0, 1, -1).astype(np.float32, copy=False)
  for row in data:
    for val in row:
      d += struct.pack('f',val)
  with open(filename,'wb') as f:
    f.write(d)

argvs = sys.argv

unit = 1000
model = L.Classifier(BinaryMLP(784, unit, 10))
chainer.serializers.load_npz(argvs[1], model)

# --- L1 -----
write_liner_W(model.predictor.l1.W.data, 'result/binary_net.l1.W.dat')
