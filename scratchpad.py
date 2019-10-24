
import heapq
from collections import defaultdict
import re
from collections import deque
# a=12
# print(int(str(a)[1]))


s = 'aabcdcb'
while s != "":
	len0 = len(s)
	ch = s[0]
	print("Before= ", s)
	s = s.replace(ch, "")
	print("after= ", s)
	len1 = len(s)
	if len1 == len0-1:
	    print(ch)
	    break;
else:
    print("no")