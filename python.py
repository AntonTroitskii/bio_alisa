import sys
import os

print('python location:')
print(os.path.dirname(sys.executable))
print('python sys')
for i in sys.path:
    print(i)
