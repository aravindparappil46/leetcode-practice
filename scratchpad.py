import math
import heapq
from collections import defaultdict
import re
from collections import deque

print(int(not 0))

nestedList = [1,2,3,4]
stack = []

for i in range(len(nestedList) - 1, -1, -1):
        stack.append(nestedList[i])

print(stack)