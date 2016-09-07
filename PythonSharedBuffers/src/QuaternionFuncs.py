import numpy as np
from numpy import sin, cos
'''quaternionFuncs-------------------------------------------------------------
Functions to create and use quaternions for robot orientation viewer
----------------------------------------------------------------------------'''
def normalize(v, tolerance = 0.00001):
  magnSqr = sum(n * n for n in v)
  if abs(magnSqr - 1.0) > tolerance:
    magn = pow(magnSqr, 1/2)
    v = tuple(n / magn for n in v)
  return v

def q_mult(q1, q2):
  w1, x1, y1, z1 = q1
  w2, x2, y2, z2 = q2
  w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
  x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
  y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
  z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
  return w, x, y, z

def q_conjugate(q):
  w, x, y, z = q
  return (w, -x, -y, -z)

def qv_mult(q1, v1):
  q2 = (0.0,) + v1
  return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]

def axisangle_to_q(v, theta):
  v = normalize(v)
  x, y, z = v
  theta /= 2
  w = cos(theta)
  x = x * sin(theta)
  y = y * sin(theta)
  z = z * sin(theta)
  return w, x, y, z

def q_to_axisangle(q):
  w, v = q[0], q[1:]
  theta = acos(w) * 2.0
  return normalize(v), theta

