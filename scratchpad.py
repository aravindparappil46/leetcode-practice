import math
import heapq
from collections import defaultdict
import re
from collections import deque


n = 3

pref = [[1,2,3,4], [3,1,4,2], [4,2,1,3]]

a = '999999999'
b = '484879996677'

longest = a if len(a) > len(b) else b

if len(a) > len(b):
	diff = len(a) - len(b)
	pad = '0'*diff
	b = str(pad) + b
else:
	diff = len(b) - len(a)
	pad = '0'*diff
	a = str(pad) + a

o = ''
for i in range(len(a)):
	i1 = int(a[i])
	i2 = int(b[i])
	o += str(i1+i2)

print(o)

 
