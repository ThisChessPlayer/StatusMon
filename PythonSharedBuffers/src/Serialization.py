from ctypes import *

# converts a ctype into a string
def Pack(ctype_instance):
    return string_at(addressof(ctype_instance), sizeof(ctype_instance))

# convert from string to ctype
def Unpack(ctype, string):
    buf = b''
    for i in string:
        buf += i.to_bytes(1, 'little')
    cstring = create_string_buffer(buf)
    ctype_instance = cast(pointer(cstring), POINTER(ctype)).contents
    return ctype_instance
