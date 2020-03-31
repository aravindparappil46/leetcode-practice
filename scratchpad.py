import math
import heapq
from collections import defaultdict
import re
from collections import deque

# arr = [2,12,13,9,10,15]
# arr = [2,19,10,1,6,13,6,6,15,12]
arr = [9,8,4,9,28,21,24,18,29,25,9,3,19,5,3]
def f(arr):
	new = [arr[0]]
	count = 1
	if arr[1] < arr[0]:
		new.insert(0,arr[1])
	elif arr[1] > arr[0]:
		new.append(arr[1])
	count += 1 

	for e in arr[2:]:
		where = bestPlace(new, e)
		print(new,"ELEM:",e,"INDEX:",where)
		if where == 0:
			count += 1
		elif where == 1:
			count += 3
		elif where == len(new):
			count += 1
		elif where == len(new)-1:
			count += 3
		else:
			count += countMid(new, where, e)
		print("Count:", count)
		new.insert(where,e)
	print(count)
	return count

def bestPlace(arr, e):
	return len([x for x in arr if x<e])

def countMid(arr,where,e):
	left = arr[:where]
	right = arr[where:]
	leftCount = 2*len(left) + 1
	rightCount = 2*len(right) + 1

	if leftCount < rightCount:
		return leftCount
	else:
		return rightCount

f(arr)


