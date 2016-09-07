from ctypes import *

class Location(Structure):
    _fields_ = [
        ("x", c_double),
        ("y", c_double),
        ("z", c_double),
        ("confidence", c_uint8),
        ("loctype", c_uint8)
    ]

class LocationArray(Structure):
    _fields_ = [
        ("locations", Location * 3)
    ]

class LocationAndRotation(Structure):
    _fields_ = [
        ("x", c_double),
        ("y", c_double),
        ("z", c_double),
        ("xrot", c_double),
        ("yrot", c_double),
        ("zrot", c_double),
        ("confidence", c_char)
    ]
