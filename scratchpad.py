import math
import heapq
from collections import defaultdict
import re
from collections import deque


n = 3

rows = [0] *n

rows[0]+= 1
rows[1] += 1
rows[2] += 1
print(n == rows[1])

print(rows)