import numpy as np
import math
import random

def setup():
  x        = random.uniform(1, 10) - 5.0
  avg_mean = random.uniform(1, 100) - 50.0
  avg_var  = random.uniform(1, 50)
  beta     = random.uniform(1, 6) - 3.0
  gamma    = random.uniform(1, 2)
  return x, avg_mean, avg_var, beta, gamma

def bn(x, avg_mean, avg_var, beta, gamma):
  x_hat = (x - avg_mean) / math.sqrt(avg_var + 0.0001)
  y = (gamma * x_hat) + beta
  return y

def bn2(x, avg_mean, avg_var, beta, gamma):
  t = avg_mean - ((beta * math.sqrt(avg_var + 0.0001)) / gamma)
  y = x - t
  return y

for i in range(0, 50):
  x, avg_mean, avg_var, beta, gamma = setup()
  print("-------")
  a = bn(x, avg_mean, avg_var, beta, gamma)
  b = bn2(x, avg_mean, avg_var, beta, gamma)
  if (a > 0.0) == (b > 0.0):
    print("OK")
  else:
    print(a,b)
