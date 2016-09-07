from ctypes import *

class Kill(Structure):
    _fields_ = [
               ("isKilled", c_bool)
               ]

class Health(Structure):
    _fields_ = [
               ("saturated", c_uint8),
               ("direction", c_uint8)
               ]

class Outputs(Structure):
    _fields_ = [
               ("motors", c_double * 8)
               ]

class PhysicalOutput(Structure):
    _fields_ = [
               ("force",  c_double * 3),
               ("torque", c_double * 3)
               ]
