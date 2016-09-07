from ctypes import *

class Linear(Structure):
    _fields_ = [
               ("pos", c_double * 3),
               ("vel", c_double * 3),
               ("acc", c_double * 3)
               ]

class Angular(Structure):
    _fields_ = [
               ("pos", c_double * 4),
               ("vel", c_double * 3),
               ("acc", c_double * 3)
               ]

class Data(Structure):
    _fields_ = [
               ("accelerometer",  c_double * 3),
               ("gyro",           c_double * 3),
               ("magnetometer",   c_double * 3),
               ("pressureSensor", c_double    ),
               ("isEnabled",      c_bool   * 4)
               ]
