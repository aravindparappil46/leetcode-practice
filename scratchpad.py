
import heapq
from collections import defaultdict
import re
from collections import deque
# a=12
# print(int(str(a)[1]))


a = [1,2,3]
def two(nums, target):
	d = {}
	out = set()
	for i in range(len(nums)-1):
		compliment = target - nums[i]
		if compliment in d:
			out.add((d[compliment], i))
		else:
			d[nums[i]] = i
	return out

print(two(a,3))