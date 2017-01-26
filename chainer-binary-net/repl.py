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
# chainer.serializers.load_npz(argvs[1], model)

# code.InteractiveConsole(globals()).interact()
