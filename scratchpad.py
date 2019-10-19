
import heapq
from collections import defaultdict
import re
from collections import deque
# a=12
# print(int(str(a)[1]))


a = [1,[2,3]]
o = []
for i in a:
	if isinstance(i,int):
		o.append(i)
	else:
		for l in i:
			o.append(l)

print(o)