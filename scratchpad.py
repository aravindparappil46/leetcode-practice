import collections
arr = ['dog', 'apple', 'bat', 'apple', 'cat', 'cat', 'bat', 'apple','cat','apple']

d = collections.Counter(arr)
a = []
for k,v in d.items():
