

import heapq

h = []

heapq.heappush(h, 2)
heapq.heappush(h, 1)
heapq.heappush(h, 99)

h[h.index(2)] = h[-1]
h.pop()
heapq.heapify(h)
print(h)