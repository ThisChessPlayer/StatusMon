# Python Shared Buffers
Shared buffer definitions in Python

# Requirements
  - ctypes

# Usage
Consider the following script
```python
from Sensor import *
from Serialization import *

linearBuffer = Linear()

for i in range(3):
    linearBuffer.pos[i] = i;
    linearBuffer.vel[i] = i + 3;
    linearBuffer.acc[i] = i + 6;

packed = Pack(linearBuffer)
unpacked = Unpack(Linear, packed)

# prints 0 through 8
for i in range(3):
    print(unpacked.pos[i])
for i in range(3):
    print(unpacked.vel[i])
for i in range(3):
    print(unpacked.acc[i])
```
Forward is x
Right is y
Down is z