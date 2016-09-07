from ctypes import *

class AxisControl(Union):
    _fields_ = [
               ("vel", c_double    ),
               ("pos", c_float  * 2)
               ]

class ControlInput(Structure):
    _fields_ = [
               ("angular", AxisControl * 3),
               ("linear",  AxisControl * 3),
               ("mode",    c_uint8        )
               ]

class Goals(Structure):
    _fields_ = [
               ("forwardVision", c_uint8),
               ("downVision",    c_uint8),
               ("sonar",         c_uint8),
               ]

class SensorReset(Structure):
    _fields_ = [
               ("pos",   c_double * 3),
               ("reset", c_bool      )
               ]
