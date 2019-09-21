import re

d = {1: 'a', 2: 's'}

try:
	print(d[3])
except:
	print('oop')

#re.sub(' +',' ',s)
s= 'apple'
d = { i:0 for i in s }
print(d)