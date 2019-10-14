import re
import collections

# Atoi
def atoi(s):
    s = s.strip()
    if s[0].isalpha() and (s[0] != '+' or s[0] != '-'):
        return 0
    if s[0] == "-":
        minus = True

    s = re.sub("\\+","",s)
    s = re.sub("\D","",s)
    if minus:
        return -int(s)

    return int(s)

# print(atoi("-1aa3"))

# Atoi no casting
def atoi2(s):
    s = s.replace(' ','')   
    sign = 1
    res = 0
    chars = "0123456789"   
    if s[0] == '-':
        sign = -1
        
    s = re.sub("\\+-","",s)
    s = re.sub("\D","",s)
    
    for i in range(len(s)):
        res += chars.index(s[i]) * 10**(len(s)-i-1)

    return res * sign

#print(atoi2("-s2q2x"))

# Add two numbers
class ListNode:
     def __init__(self, x):
         self.val = x
         self.next = None

class Solution:
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        return self.convert_listnode_to_num(l1) + self.convert_listnode_to_num(l2)

    def convert_listnode_to_num(listnode):
        digit = 0
        num = 0

        while listnode is not None:
            num += listnode.val * (10 ** digit)
            digit += 1
            listnode = listnode.next
        return num

    def convert_num_to_listnode(num):
        num_str = str(num)

        node_list = []
        prev_node = None
        for char in reversed(num_str):
            node = ListNode(int(char))
            node_list.append(node)
            if prev_node is not None:
                prev_node.next = node
            prev_node = node
            
        return node_list[0]

if __name__ == '__main__':
    l1 = Solution.convert_num_to_listnode(342)
    l2 = Solution.convert_num_to_listnode(465)
    #print(Solution().addTwoNumbers(l1, l2))


# Length of Longest substring without repeating characters
def lengthOfLongestSubstring(s):
    lastRepeating = -1
    longestSubstring = 0
    positions = {}
    
    for i in range(0, len(s)):
        if s[i] in positions and lastRepeating < positions[s[i]]:
            lastRepeating = positions[s[i]]
            
        if i - lastRepeating > longestSubstring:
            longestSubstring = i - lastRepeating
            
        positions[s[i]] = i
    print(positions)
    return longestSubstring

# print(lengthOfLongestSubstring("bbbbbam"))

# Longest palindromic substring
def longestPalindrome(s):
    longest = ""
    for i in range(len(s)):
        temp = expandFromCenter(s, i, i)
        print(i,i,temp)
        if len(temp) > len(longest):
            longest = temp
            
        temp = expandFromCenter(s, i, i+1)
        print('===',i, i+1, temp)
        if len(temp) > len(longest):
            longest = temp
            
    return longest

def expandFromCenter(s, start, end):
    while start >= 0 and end <= len(s)-1 and s[start] == s[end]:
        start -= 1
        end += 1
    return s[start+1:end]

# print(longestPalindrome("abxbax"))


# 3 Sum
# Time complexity = O(n^2)
def threeSum(arr, t):
    arr.sort()
    result = set() # not a list, to avoid duplicates
    r = len(arr)-1
    
    for i in range(len(arr)):
        l = i + 1
        while(l < r):
            s = arr[i] + arr[l] + arr[r]
            if s < t:
                l += 1
            if s > t:
                r -= 1
            if s == t:
                result.add((arr[i], arr[l], arr[r]))
                l += 1
                
    return result

arr = [0,8,1,9,22,2]
# print(threeSum(arr,9))
# print(threeSum([-1, 0, 1, 2, -1, -4], -3))


# Letter combinations of phone dial
def letterCombinations(digits):
    phone = {
                '2': ['a', 'b', 'c'],
                 '3': ['d', 'e', 'f'],
                 '4': ['g', 'h', 'i'],
                 '5': ['j', 'k', 'l'],
                 '6': ['m', 'n', 'o'],
                 '7': ['p', 'q', 'r', 's'],
                 '8': ['t', 'u', 'v'],
                 '9': ['w', 'x', 'y', 'z']
        }
    output = []
    
    if digits:
        recurse(output,'',digits, phone)

    print(output)

def recurse(output, combo, digits, phone):
    if len(digits) == 0:
        output.append(combo)

    else:
        for letter in phone[digits[0]]:
            recurse(output, combo + letter, digits[1:], phone)


#letterCombinations('2398')

# Reverse a number

def reverse(x: 'int') -> 'int':
    rev = 0
    while x > 0:
        pop = x % 10
        x = x // 10
        rev = rev * 10 + pop
    
    return rev

#print(reverse(21))

# Roman to integer
def romanToInt(s):
    rom_val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    int_val = 0
    for i in range(len(s)):
        if i > 0 and rom_val[s[i]] > rom_val[s[i - 1]]:
            int_val += rom_val[s[i]] - 2 * rom_val[s[i - 1]]
        else:
            int_val += rom_val[s[i]]
    return int_val

#print(romanToInt("MMMM"))

# Merge two sorted arrays
# Time complexity = O(n1 + n2) (Space also)
def mergeSortedArrays(a,b):
    c = [None]*(len(a)+len(b))
    i=0
    j=0
    k=0
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            c[k] = a[i]
            k+=1
            i+=1
        else:
            c[k] = b[j]
            k+=1
            j+=1

    while i < len(a):
        c[k] = a[i]
        k+=1
        i+=1
    while j < len(b):
        c[k] = b[j]
        k+=1
        j+=1

    return c

a = [1,2]
b = [5,6,7]
#print(mergeSortedArrays(a,b))

# Merge k sorted arrays
# [[1,2], [55,44,33], [10,11,87]]
import heapq
def mergeKSortedArrays(arr):
    heap = []
    for array in arr:
        for element in array:
            heapq.heappush(heap, element)
    while len(heap) > 0:
        print(heapq.heappop(heap))

arr=[[1,2], [33,44,55], [10,11,87]]
#mergeKSortedArrays(arr)

# Merge K Sorted Arrays in O(n*logk) where n is total num of elements
# Create a heap
# First, add 1st elements of all arrays into it
# Now, popping from heap will give least so far..keep track of the array from which it came from
# Add this popped val to final result array, as it is the smallest ele so far
# Remove this popped val from its original array and 
# push the new top element of THAT array into the heap..Repeat till heap is empty
#
#
# pop and push for minheap - O(logk)
# Time complexity - O(n*log(k)) where n is total num of elements 
def mergeKSortedArrays_better(arr):
    heap = []
    output = []

    # Using a tuple to keep track of which array the elements belong to
    for index, array in enumerate(arr):
        heapq.heappush(heap, (array[0], index)) 

    while len(heap) > 0:
        smallest = heapq.heappop(heap)

        idx_of_arr_it_came_from = smallest[1] #coz we now need to add to heap from THAT array
        smallest_value = smallest[0]

        output.append(smallest_value)

        arr[idx_of_arr_it_came_from].pop(arr[idx_of_arr_it_came_from].index(smallest_value))

        if len(arr[idx_of_arr_it_came_from]) > 0:
            heapq.heappush(heap, (arr[idx_of_arr_it_came_from][0], idx_of_arr_it_came_from))

    print(output)

#mergeKSortedArrays_better(arr)


# Remove duplicates from array
def removeDup(a):
    return list((collections.Counter(a)).keys())

a = [1,1,2,1,1,1,2,3,4,4,4,4,4,4]
#print(removeDup(a))


# strStr()
def strStr(haystack, needle):
    i = 0
    while i <= len(haystack) - len(needle):
        if haystack[i:i+len(needle)] == needle:
            return i
        i += 1
    return -1

#print(strStr("apple","l"))

# Group anagrams
from collections import defaultdict

def groupAnagrams(inp):
    d = defaultdict(list)
    for word in inp:
        d[''.join(sorted(word))].append(word)
    return list(d.values())

inp = ['tea','eat','aet','cat','tac','mat']
#print(groupAnagrams(inp))


# is same tree
def isSameTree(p,q):
    if not p and not q:
        return True
    if not q or not p:
        return False
    if p.val != q.val:
        return False
    
    return isSameTree(p.right, q.right) and \
           isSameTree(p.left, q.left)


# Symmetric tree
"""
def isSymmetricRoot(r):
    if not r:
        return True
    return isSymmetric(r.left, r.right)

def isSymmetric(l,r):
    if not l and not r:
        return True
    if not l or not r:
        return False
    if l.val != r.val:
        return False

   return isSymmetric(l.left, r.right) and isSymmetric(l.right, r.left)
   """

# Two Sum
def twoSum(a, target):
    d = {};
    o = set()
    for i in range(len(a)):
        compliment = target - a[i]
        if compliment in d:
            o.add(( d[compliment], i ))
        d[a[i]] = i 
    return o

# THIS IS THE RIGHT ANSWER!
def twoSum1(a, target):
    o = set()
    d= {}
    for i in range(len(a)):
        compliment = target - a[i]
        if compliment in d:
            temp = [compliment, a[i]]
            temp.sort()
            if str(temp) not in o:
                o.add(str(temp))
        d[a[i]] = i
    print(d)
        
    return o

a1 = [6,6,3,9,3,5,12]
a = [7,8,13,20,12,12,6,1,11]
print(twoSum1(a1,12))

# max depth of tree
def maxDepth(node): 
    if node is None: 
        return 0
    else: 
        lDepth = maxDepth(node.left) 
        rDepth = maxDepth(node.right) 
  
        if lDepth > rDepth: 
            return lDepth+1
        else: 
            return rDepth+1

# Sorted array to BST
def sortedArrayToBST(arr): 
    if not arr: 
        return None
  
    mid = (len(arr)) / 2
    # make the middle element the root 
    root = Node(arr[mid]) 
    root.left = sortedArrayToBST(arr[:mid]) 
    root.right = sortedArrayToBST(arr[mid+1:])
    
    return root 


# Remove duplicates from array
def removeDup(a):
    d = {i:0 for i in a}
    for key, val in d.items():
        print(key)
        
a = [1,1,2,1,1,1,2,3,4,4,4,4,4,4]
#removeDup(a)

# Remove duplicates unnecessary complications :P
def removeDup2(a):
    new = []    
    for i in range(len(a)):
        if a[i] not in new:
            new.append(a[i])

    return new

#print(removeDup2(a))


# Palindrome Partitioning
# Find all substrings and remove non palindromes
def partition(s:str):
    perms = []
    for i in range(len(s)):
        for end in range(i+1, len(s)+1):
            substring_to_check = s[i: end]
            if palindrome(substring_to_check):
                if substring_to_check not in perms:
                    perms.append(substring_to_check)

    print(perms)

def palindrome(s):
    return True if s == s[len(s)::-1] else False
     
#partition('malayalam')

# LRU Cache
# from collections import defaultdict
# I think this is LFU Hmmm????????
class LRUCache:

    def __init__(self, capacity: int):
        self.data = defaultdict(list)
        self.max_cap = capacity

    def get(self, key: int) -> int:
        try:
            if key not in self.data.keys():
                return -1
            self.data[key][1] += 1
            return self.data[key][0]
        except Exception as e:
            print(e)
            return -1

    def put(self, key: int, value: int) -> None:
        key_to_replace = self.check_and_replace()
        if key_to_replace is not None:
            del self.data[key_to_replace] 
        self.data[key] = [value, 1]

    def check_and_replace(self):
        if len(self.data) >= self.max_cap:
            least_count_so_far = list(self.data.values())[0][1]
            key_to_replace = list(self.data.keys())[0]
            for k,v in self.data.items():
                if v[1] < least_count_so_far:
                    least_count_so_far = v[1]
                    key_to_replace = k
            return key_to_replace

        else:
            return None

# new1 = LRUCache(2)
# print(new1)
# print(new1.put(1,1))
# print(new1.put(2,2))
# print(new1.get(1))
# print(new1.put(3,3))
# print(new1.get(2))
# print(new1.put(4,4))
# print(new1.data)
# print(new1.get(1))
# print(new1.get(3))
# print(new1.get(4))

# Reverse words in string
# hello world --> world hello
# Remove multiple spaces

def reverseWords(s):
    s = re.sub(' +', ' ', s).strip()
    arr = s.split(' ')
    for word in arr[::-1]:
        print(word, end=" ")

#reverseWords('  hello    world!   s')

# Bloomberg questions
"""
1. Comma formatting 104450 -> 104,450
100000 -> 10,00,000 and 1,000,000

2. Given integer, # ways it can be represented as sum of 1s and 2s
3. Reverse sentence in place
4. Given array of ints and an int, find avg. of m groups in O(n)
5. Given sorted array, find first occurrence of given number
6. Brackets matching
7. Given 2 strings, find missing string ("This is bad", "is bad" => This)
8. Given pair of ranges, merge overlapping pairs
9. LCA in BST. Function takes two node ptrs
10. Sum of nums represented as LLs
11. Given stream of nums & terminator, return first unique # till terminator
12. Run length encoding. aaaaabb -> a5b2
13. Given binary tree, test whether BST
14. Move all zeroes to end of array
15. Longest substring with unique characters O(n)
16. All nodes matching given value in a tree
17. All prime numbers in a range
18. Freq of occurrence of words in array, sort em. Find top k elements (heap)
19. Print level at which node is in a binary tree
20. Iterative fibonacci
21. Find anagrams from array of strings
22. Flatten singly LL (Multilevel LL ??????) <<<IMP
23. Three Sum <<<IMP
24. Given string, insert spaces after words, given dict of valid words
25. Level order traversal of binary tree
26. One edit distance
27. All numbers and characters in a string must be sorted and placed on the indexes of char only.  
28. Deep copy of graph <<<IMP
29. Deep copy linked list with random and next ptr <<<IMP
30. Find element in rotated sorted array <<<IMP
31. Campus bikes 
32. Campus bikes II ????????
33. Find kth largest node in BST <<<IMP
34. Given a number of children, and a number of turns for each round. Figure out which student wins  <<<IMPT
35. Best time to sell stocks <<<IMP
36. LRU Cache <<<IMP
"""

# 1. Comma formatting -Indian and US
# Indian - 17,77,77,789
# US - 177,777,789
from collections import deque
def comma(n, locale):
    stack = deque()
    result = ''

    for digit in str(n):
        stack.append(digit)

    if locale == 'IN':
        for i in range(len(stack)):
            if i % 2 != 0 and i != 1:
                result += ','
            result += stack.pop()

    elif locale == 'US':
        for i in range(len(stack)):
            if i % 3 == 0 and i != 0:
                result += ','
            result += stack.pop()

    print(result[::-1])

n = 177777789
# comma(n, 'US')


# 2. No. of ways integer can be represented as sum of 1s and 2s
#
# Answer will be the n+1 Fibonacci number
# DP[n] be num of ways to write N as sum of 1 and 2
# N = x1 + x2 + x3 + ...+xn
# If last num is 1, then sum of remaining nums is n-1
def sumOf1sAnd2s(n):
    DP = [0 for i in range(0, n + 1)] 
    # base cases 
    DP[0] = DP[1] = 1
  
    # Iterate for all values from 2 to n 
    for i in range(2, n + 1): 
        DP[i] = DP[i - 1] + DP[i - 2]
      
    return DP[n] 

# print(sumOf1sAnd2s(5))


# 3. Reverse a string in place 
# Strings are immutable in python. But in C, they are array of chars

def reverseInPlace(s):
    arr = []
    arr.extend(s) # ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o'...]
    for i in range(0, int(len(arr)/2)):
        arr[i], arr[len(arr)-i-1] = arr[len(arr)-i-1], arr[i]
    print(''.join(arr))

#reverseInPlace('hello world') # dlrow olleh 

def reverseInPlace2(s):
    s = re.sub(' +', ' ', s).strip()
    return s[::-1]
#print(reverseInPlace2('hello world')) # dlrow olleh 

def reverseWords2(s):
     s = re.sub(' +', ' ', s).strip()
     stack = s.split()
     for i in range(len(stack)):
        print(stack.pop(), end=" ")

#reverseWords2('hello world') # world hello

# 4. Given arry of ints and int m, find avg of m groups
# Maximum avg. sub array
# Keep another array for cumulative sum
# Max avg will be sub arry with max sum. Avoids floating pt calcs.
#
# Returns start index of subarray
def avgOfGroups(arr, m):
    cumulative_sum = [0] * len(arr)
    cumulative_sum[0] = arr[0]

    if m > len(arr):
        return -1

    for i in range(1, len(arr)):
        cumulative_sum[i] = cumulative_sum[i-1] + arr[i]

    # print(arr)
    print(cumulative_sum)
    max_so_far = cumulative_sum[0]
    max_end = 0

    for i in range(m, len(arr)):
        curr_sum = cumulative_sum[i] - cumulative_sum[i-m]
        if curr_sum > max_so_far:
            max_so_far = curr_sum
            max_end = i

    print('max_end ',max_end)
    return max_end - m + 1   

arr = [1,2,4,7,1,2,5]
m = 4
#print(avgOfGroups(arr, m))

# 5. First occurence of given number in sorted array
# Binary search modified to go left when match found
def firstOccurence(arr, t):
    left = 0
    right = len(arr) - 1
    result = 0

    while left <= right:
        mid = int((left + right) / 2)
        if arr[mid] < t:
            left = mid + 1
        elif arr[mid] > t:
            right = mid - 1
        elif arr[mid] == t:
            result = mid
            right = mid - 1 # Go left to find any other occurence. Change to left = mid + 1 for last occurence

    return result

arr = [1,2,3,3,4,4,5,5,6,7,8]
t = 3
#print(firstOccurence(arr,t))

# 6. Brackets matching - paranthesis matching
def brackets(s):
    stack = deque()
    openings = ['(', '{', '[']
    for e in s:
        if e in openings:
            stack.append(e)
        else:
            try:
                bracket_to_check = stack.pop()
                if (e == ')' and bracket_to_check != '(') or \
                   (e == '}' and bracket_to_check != '{') or \
                   (e == ']' and bracket_to_check != '['):
                    return False

            except Exception as e:
                return False
    return True


s = '({}[)]'
#print(brackets(s))

# 7. Given 2 strings, find missing word
# Difference between strings
# Get the longest string and mark all as 1 in a dict
# Iterate thru shorter string and mark as -1 if present in dict
# Words without -1 are missing from shorter string
def missingWord(s1, s2):
    d = {}
    s1 = re.sub(' +', ' ', s1).strip()
    s2 = re.sub(' +', ' ', s2).strip()
    if len(s1) > len(s2):
        longest_str = s1
        second_str = s2
    else:
        longest_str = s2
        second_str = s1

    for words in longest_str.split():
        d[words] = False

    for check_words in second_str.split():  
        if not d[check_words]:
            d[check_words] = True # common in both strs
    
    # Uncommon will be False
    for key, value in d.items():
        if not value:
            print(key, end= " ")

s1 = 'this sure is a cat'
s2 = 'this is'
#missingWord(s1,s2)

# 8. Merge overlapping ranges intervals
# Sort by starting time, push to stack, compare with Top-of-stack
# O(nlogn) considering the sorting. Else O(n)
def mergeIntervals(arr):
    stack = deque()
    arr = sorted(arr, key =lambda a: a[0])
    stack.append(arr[0])

    for pair in arr:
        start = pair[0]
        end = pair[1]
        tos_end = stack[-1][1]
        tos_start = stack[-1][0]
        if tos_end < end and start < tos_end:
            stack.pop()
            stack.append((tos_start, end))
        elif tos_end < start:
            stack.append(pair)

    for pairs in stack:
        print(pairs)

arr = [(10,11), (6,9), (1,3), (2,5), (7,8)]
# mergeIntervals(arr)

# 9. LCA in BST
# If root is > n1 and n2, go left
# If root is < n1 and n2, go right

class Node:
    def __init__(self, v):
        self.data = v
        self.left = None
        self.right = None

def lca(root, n1, n2):
    if root.data > n1 and root.data > n2:
        return lca(root.left, n1, n2)
    elif root.data < n1 and root.data < n2:
        return lca(root.right, n1, n2)
    return root

root = Node(20) 
root.left = Node(8) 
root.right = Node(22) 
root.left.left = Node(4) 
root.left.right = Node(12) 
root.left.right.left = Node(10) 
root.left.right.right = Node(14) 
  
n1 = 10 ; n2 = 14
#res = lca(root, n1, n2)
#print(res.data)


# 10. Sum of two numbers represented as Linked Lists
class Node:
     def __init__(self, x):
         self.val = x
         self.next = None

class LLSum:
    def addTwoNumbers(self,node1, node2):
        """
        :type l1: Node
        :type l2: Node
        :rtype: Node
        """
        return self.convert_node_to_num(node1) + self.convert_node_to_num(node2)

    def convert_node_to_num(self, node):
        digit = 0 # represent tens, hundreds, thousands
        num = 0
        while node is not None:
            num += node.val * (10 ** digit)
            digit += 1
            node = node.next
        return num

    def convert_num_to_node(self, num):
        num_str = str(num)

        node_list = []
        prev_node = None
        for char in reversed(num_str):
            node = Node(int(char))
            node_list.append(node)
            if prev_node is not None:
                prev_node.next = node
            prev_node = node
            
        return node_list[0]

if __name__ == '__main__':
    obj = LLSum()
    node1 = obj.convert_num_to_node(342)
    node2 = obj.convert_num_to_node(465)
    #print(obj.addTwoNumbers(node1, node2))

# 11. Given stream of numbers, find unique numbers till that point
# Find first non repeating character in stream
def uniqueInStream(s):
    inDLL = []* 256
    repeated = [False] * 256
    for i in range(len(s)):
        x = s[i]
        if not repeated[ord(x)]: # not repeated
            if x not in inDLL:
                inDLL.append(x)
            else: # x already seen...remove it from DLL and mark as repeated
                inDLL.remove(x)
                repeated[ord(x)] = True
        if(len(inDLL) > 0):
            print('First non repeating char so far: ', inDLL[0])

s = 'thisisthisah'
#uniqueInStream(s)

# 12. Run length encoding
def runlengthEncoding(s):
    d = { i:0 for i in s }
    output = ''
    for i in s:
        d[i] += 1
    for k,v in d.items():
        output += k + str(v)
    return output

s = 'apple'
# print(runlengthEncoding(s))

# 13. Check if given tree is valid BST
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def helper(node, min, max):
    if not node:
        return True

    if node.val <= min or node.val >= max:
        return False

    if not helper(node.left, min, node.val):
        return False

    if not helper(node.right, node.val, max):
        return False

    return True

def isBST(root):
    return(helper(root, float('-inf'), float('inf')))

root = TreeNode(4)
root.left = TreeNode(2) 
root.right = TreeNode(5) 
root.left.left = TreeNode(1) 
root.left.right = TreeNode(3) 

#print(isBST(root))


# 14. Move all zeroes to end of array
# count variable keeps track of non-zero nums..swap iterator with count
# to move non-zeroes ahead
# Time - O(n)
# Space - O(1)
def moveZeroesToEnd(arr):
    count = 0
    for i in range(len(arr)):
        if arr[i] != 0:
            arr[i], arr[count] = arr[count], arr[i]
            count+=1
    return arr

arr = [1, 2, 7, 0, 8, 6, 3, 0, 8, 87, 1, 2, 0, 2, 0, 2, 0, 0]
# print(moveZeroesToEnd(arr))


# 15. Longest substring with unique characters
# See line 85 for length of longest Unique Substring 
def longestUniqueSubstring(s):
    max_len = 0  # max substring len so far 
    curr_st = 0 # start index of curr substring
    start = 0
    d = {} # holds the last repeated index for each char
    d[s[0]] = 0

    for i in range(1, len(s)):
        if s[i] not in d: # Char not seen before...add curr index to dict
            d[s[i]] = i

        else:
            # check if occurrence is before or after starting of curr substring
            if d[s[i]] >= curr_st:
                curr_len = i - curr_st

                if curr_len > max_len:
                    max_len = curr_len
                    start = curr_st
                
                curr_st = d[s[i]] + 1 # Next substring starts from last occurrence
            d[s[i]] = i # updating the last occurence of this char

    if i - curr_st > max_len :  # Comparing len of last substring with max so far
        max_len = i - curr_st
        start = curr_st

    return s[start : start + max_len]

s = 'apple'
#print(longestUniqueSubstring(s))

"""
input:
factual-commons => [apache-commons, guava, thrift]
map-reduce => [apache-commons, hadoop]
place-attach => [factual-commons, map-reduce]
hive => [hadoop, apache-commons]
hive-querier => [hive, factual-commons]

output:
hive-querier => [hadoop, apache-commons, hive, guava, thrift, factual-commons, hive-querier]
"""
deps_dict = {
    "factual-commons" : ['apache-commons', 'guava', 'thrift'],
    "map-reduce" : ['apache-commons', 'hadoop'],
    "place-attach" : ['factual-commons', "map-reduce"],
    'hive': ['hadoop', 'apache-commons'],
    'hive-querier': ['hive', 'factual-commons']
}
to_build = 'hive-querier'

def minimumDependencies(deps_dict, to_build):
    built = []
    for pkg in deps_dict[to_build]:
        unadded_pkgs = returnNewDeps(pkg, built)
        if len(unadded_pkgs) > 0:
            built += [i for i in unadded_pkgs]

    built.append(to_build)
    print(built)

def returnNewDeps(check, built):
    new_deps = []
    
    if check not in deps_dict:
        return []

    for i in deps_dict[check]:
        if i not in built:
            new_deps.append(i)
    new_deps.append(check)
    return new_deps

#minimumDependencies(deps_dict, to_build)


# Factual -- time and speed question
def solution(readings, end_time):
    # Type your solution here
    dist_so_far = 0
    curr_speed = 0
    last_time = 0
    for i in range(len(readings)):
        if len(readings) > 1:
            time_diff = abs(readings[i+1][0] - readings[i][0]) / 3600
            curr_speed = readings[i][1]
            dist_so_far += curr_speed * time_diff
            last_time = readings[i][0]
        else:
            last_time = readings[i][0]
            curr_speed = readings[i][1]


    r = abs((last_time - end_time)) /3600
    print('real', curr_speed * r)

a = [[0,90],[300,80]]


# 16. Search for a node in a tree

class Node:
    def __init__(self, val):
        self.data = val
        self.right = None
        self.left = None

def doesKeyExist(root, key):
    if root == None:
        return False

    if root.data == key:
        return True
    return doesKeyExist(root.left, key) or doesKeyExist(root.right, key)

root = Node(0)
root.left = Node(1)  
root.left.left = Node(3)  
root.left.left.left = Node(7)  
root.left.right = Node(4)  
root.left.right.left = Node(8)  
root.left.right.right = Node(9)  
root.right = Node(2)  
root.right.left = Node(5)  
root.right.right = Node(6)  

#print(doesKeyExist(root, 11))


# 17. Prime numbers within a range
# Given, start and end - find all primes within those nums

def primeNumbersWithinRange(start, end):
    primes = []
    for i in range(start, end):
        if i == 1:
            continue
        if isPrime(i):
            primes.append(i)
    return primes


def isPrime(n):
    for i in range(2, n):
        if n % i == 0:
            return False
    return True

#print(primeNumbersWithinRange(10,20))

# 18. Frequency of occurence of words in an array, sort them. Find top k, bottom k
#
# O(nlogn) approach
def freqOfWords(arr, k):
    d = {}
    for word in arr:
        if word in d:
            d[word] += 1
        else:
            d[word] = 1

    #sorting..takes O(nlogn)
    sorted_arr = []
    for key, v in sorted(d.items(), key=lambda a: a[1], reverse = True):
        sorted_arr.append(key)

    # print(d)
    # print(sorted_arr)
    # Finding top k and bottom k
    print("Top", k, sorted_arr[:k])
    print("Bottom", k, sorted_arr[-k:])

# Using priority queue/heap
# heapify - O(N log(k)) where N is number of elements in arr
# heappop/extractMin - O(log k)
# heappush - O(log k)
# getMin - O(1) because we do heap[0] and it doesn't pop.
# 
# Total time complexity = O(nlogk)
import heapq
def freqOfWordsHeap(arr, k):
    d = {}
    for word in arr:
        if word in d:
            d[word] += 1
        else:
            d[word] = 1

    freq_min_heap = []
    freq_max_heap = []
    
    for key, v in d.items():
        heapq.heappush(freq_min_heap, (v,key))
        heapq.heappush(freq_max_heap, (-v,key))

    print(k, "least frequently occuring words")
    for i in range(0, k):
        print(heapq.heappop(freq_min_heap))

    print(k, "most frequently occuring words")
    for i in range(0, k):
        print(heapq.heappop(freq_max_heap))
    
arr = ['dog', 'apple', 'bat', 'apple', 'cat', 'cat', 'bat', 'apple','cat','apple','man']
# freqOfWords(arr,2)
#freqOfWordsHeap(arr,2)

# 19. Level at which a node is present in binary tree
# Level starts at 1...root is at 1
#
# Time complexity = O(n) where n is num of nodes in tree
class Node:
    def __init__(self, value):
        self.data = value
        self.left = None
        self.right = None

def getLevelHelper(node, key, level):
    if node is None:
        return 0

    if node.data == key:
        return level

    l = getLevelHelper(node.left, key, level + 1)
    if l != 0: # If 0, then it means we reached a leaf
        return l
    else:
        return getLevelHelper(node.right, key, level+1)

def getLevel(root, key):
    return getLevelHelper(root, key, 1)

root = Node(1) 
root.left = Node(2) 
root.right = Node(3) 
root.left.left = Node(4) 
root.left.right = Node(5) 

#print(getLevel(root, 5))


# 25. Level order traversal
# Create a queue and push the root
# Print its value and repeat for left child and right child
# till queue is empty
# BFS
# Time complexity = O(n)
class Node:
    def __init__(self, value):
        self.data = value
        self.left = None
        self.right = None

def levelOrderTraversal(root):
    if root is None:
        return -1

    queue = []
    queue.append(root)
    while len(queue) > 0:
        node = queue.pop(0)
        
        print(node.data)
        if node.left is not None:
            queue.append(node.left)
        if node.right is not None:
            queue.append(node.right)

root = Node(1) 
root.left = Node(2) 
root.right = Node(3) 
root.left.left = Node(4) 
root.left.right = Node(5) 

#levelOrderTraversal(root)

# 20. Iterative Fibonacci
# 1,1,2,3,5,8 (first fib num is 1)
def fibonacci(n):
    if n == 0 or n == 1:
        return 1
    else:
        x = 1
        y = 1

        for i in range(2, n):
            z = x + y
            x = y 
            y = z
        return y

#print(fibonacci(4))

# 21. Find anagrams from array of strings
def anagrams(arr):
    d = defaultdict(list)
    for word in arr:
        afterSort = ''.join(sorted(word))
        if word not in d[afterSort]:
            d[afterSort].append(word)

    for val in d.values():
        if len(val) > 1:
            print(val)

arr = ['tac', 'cat', 'act', 'blaaa','aaalb', 'sow','woe']
# anagrams(arr)

# 22. Flatten singly linked list
# 15 -> 20 -> 22 -> 30

class Node:
    def __init__(self, val):
        self.data = val
        self.next = None

def flatten(head):
    out = []
    while head.next is not None:
        out.append(head.data)
        head = head.next

    out.append(head.data)
    print(out)

node = Node(5)
node.next = Node(20)
node.next.next = Node(100)
node.next.next.next = Node(9)

#flatten(node)

# 1---2--3---4---5---6--NULL
#        |
#        7---8---9---10--NULL
#            |
#            11--12--NULL

# 1-2-3-7-8-11-12-9-10-4-5-6-NULL
class Node:
    def __init__(self, val):
        self.data = val
        self.next = None
        self.prev = None
        self.child = None

# def flattenMultiLevel(head):
#     # Hmmmmmmm????????

# 23. Three Sum
def threeSumm(arr,t):
    arr.sort()
    out = set()
    r = len(arr) - 1
    for i in range(len(arr)):
        l = i + 1
        while l < r:
            s = arr[i] + arr[l] + arr[r]
            if s < t:
                l += 1
            if s > t:
                r -= 1
            if s == t:
                out.add((arr[i], arr[l], arr[r]))
                l += 1

    if len(out) > 0:
        return out
    else:
        return None

arr = [0,8,1,9,22,2]
# print(threeSumm(arr,9))

# 24. Given string, insert spaces after words, given dict of valid words
def insertSpaces(s, validWords):
    temp = ''
    result = ''
    validWords = list(map(lambda x: x.lower(), validWords))
    for letter in s:
        temp += letter
        if temp.lower() in validWords:
            result = result + temp + ' '
            temp = ''

    if len(result) > 0:
        print(result.rstrip())
    else:
        print("No valid words found for", s)
    
   
s = 'iiiiiiiii'
valid = ['i', 'like', 'sam', 'sung', 'samsung', 'mobile', 'ice', \
  'cream', 'icecream', 'man', 'and','go', 'mango']
# insertSpaces(s, valid)


# Clutter
# p = ["an apple", "apple boy", "apple sux"]
# output should be ['an apple boy', 'an apple sux']
def generatePhrases(phrases):
    d = defaultdict(list)
    all_ending_words = set(map(lambda x: x.split()[-1], phrases))
    print(all_ending_words)
    for p in phrases:
        if p.split()[0] in all_ending_words:
            d[p.split()[0]].append(p)
    out = []
    for p in phrases:
        starting_word = p.split()[-1]
        if starting_word in d:
            for sentence in d[starting_word]:
                out.append(p +' '+sentence.split(' ',1)[1])

    print(out)
            
phrases = ["an apple", "apple boy", "apple sux"]
# generatePhrases(phrases)


# 31. Campus bikes I - Assign closest guy to bike
def campusBikes(workers, bikes):
    heap = []
    out = []
    bike_occupied = [False] * len(bikes)
    worker_got_bike = [False] * len(workers) 

    for worker_index, worker in enumerate(workers):
        for bike_index, bike in enumerate(bikes):
            d = manhattan(worker, bike)
            heapq.heappush( heap, (d, worker_index, bike_index) )

    while len(heap) > 0:
        _, worker_index, bike_index = heapq.heappop(heap)

        if not bike_occupied[bike_index]:
            if not worker_got_bike[worker_index]:
                bike_occupied[bike_index] = True
                worker_got_bike[worker_index] = True
                print("Worker", worker_index, "gets", "Bike", bike_index)
                out.append(bike_index)

    print(out)

def manhattan(w, b):
    return abs(w[0] - b[0]) + abs(w[1] - b[1])

workers = [[0,0], [1,1], [2,0]]
bikes = [[1,0], [2,2], [2,1]]
# out shud be [0, 2, 1] (for part 1)
# out shub be 4

# workers = [[0, 0], [2, 1]]
# bikes = [[1, 2], [3, 3]]
# out shub be [1, 0] (for part 1)
# out shud be 6 (part 2 min dist)
# campusBikes(workers, bikes)

# 32. Campus Bikes II - Assign bikes such that total manhattan distance is min
def campusBikes2(workers, bikes):
    # Hmmmmmmm ????????
    heap = []
    bike_occupied = [False] * len(bikes)
    worker_got_bike = [False] * len(workers) 
    num_of_workers = len(workers)

    for worker_index, worker in enumerate(workers):
        for bike_index, bike in enumerate(bikes):
            dist = manhattan(worker, bike)
            heapq.heappush( heap, (dist, worker_index, bike_index) )

    print(heap)


def isUniqueWorkerAndBike(a, b):
    if a[0] == b[0] or a[1] == b[1]:
        return False
    return True

# campusBikes2(workers, bikes)

# 30. Search in Rotated sorted array
# Time complexity O(log n)
# [0,1,2,4,5,6,7]
# [4,5,6,7,0,1,2]
#
def searchRotatedSortedArray(arr, t):
    l = 0
    r = len(arr) - 1

    while l <= r:
        mid = (l+r) // 2 # instead of r, can do (r-l)
        if arr[mid] == t:
            return mid

        if arr[mid] >= arr[l]: # left half is sorted
            if t < arr[mid] and t >= arr[l]:
                r = mid -1
            else:
                l = mid + 1

        if arr[mid] <= arr[r]: # right half is sorted
            if t > arr[mid] and t <= arr[r]:
                l = mid + 1
            else:
                r = mid + 1
    return -1

arr = [4,5,6,7,0,1,2]
# print(searchRotatedSortedArray(arr, 2))


# Two city scheduling
# Sort by priceA - priceB (this is cheapest to send to A)
# Send first len(arr)/2 ppl to city A and the other arr/2 to city B
def twoCityScheduling(arr):
    result = 0
    sorted_arr = sorted(arr, key=lambda x: x[0] - x[1])
   
    for i in sorted_arr[:len(arr)//2]:
        result += i[0]

    for i in sorted_arr[len(arr)//2:]:
        result += i[1]

    print(result)

arr = [[10,20],[30,200],[400,50],[30,20]] # out shud be 110
# twoCityScheduling(arr)


# BAAAT => BT
# AAATmmmA => TA
# Push first letter to stack
# Start a loop from second letter till end of string
# if curr char == top of stack, increment a count and push to stack, 
# if not, reset count and push to stack
# if count == 3, pop everything from stack
# Whatever is remaining in stack is the answer
def crushConsecutive(st):
    result = ''
    stack = []
    stack.append(s[0])
    count = 1 

    for i in range(1, len(st)):
        if len(stack) > 0: # this is for when there are 3 consec at the starting itself
            if st[i] == stack[-1]:
                count += 1
                stack.append(st[i])

                if count == 3: # this decides how many is the limit
                    stack.pop()
                    stack.pop()
                    stack.pop()
            else:
                count = 1
                stack.append(st[i])

        else:
            stack.append(st[i])

    print(''.join(stack))

s = "AAATmmmA"
# crushConsecutive(s)

# Candy Crush
# Stable means no more candies to crush

def candyCrush(board):
    R = len(board)
    C = len(board[0])

    while not isStable(board):
        board = crush(board)

    return board
       
# Is the board un-crushable anymore?
def isStable(board):
    R = len(board)
    C = len(board[0])

    if shouldGravityBeInvoked(board):
        return False

    # checking horizontals
    for r in range(R):
        for c in range(C-2):
            if (abs(board[r][c]) == abs(board[r][c+1]) == abs(board[r][c+2])) and \
               (abs(board[r][c]) == abs(board[r][c+1]) == abs(board[r][c+2]) != 0):
                return False

    # checking verticals
    for r in range(R-2):
        for c in range(C):
            if (abs(board[r][c]) == abs(board[r+1][c]) == abs(board[r+2][c])) and \
               (abs(board[r][c]) == abs(board[r+1][c]) == abs(board[r+2][c])!= 0):
                return False

    return True

def shouldGravityBeInvoked(board):
    R = len(board)
    C = len(board[0])
    for r in range(R):
        for c in range(C):
            # If there's a 0 in any row other than 1st row
            if board[r][c] == 0 and r != 0 and \
               board[r-1][c] != 0: # This condition is when all cells are 0s in a col
                return True
    return False

# If there's a cell with 0, move down the cell above it
# Unless we are at the first zero (nothing above it to move down)
def invokeGravity(b):
    R = len(b)
    C = len(b[0])
    for r in range(R):
        for c in range(C):
            if b[r][c] == 0 and r != 0:
                b[r][c] = b[r-1][c]
                b[r-1][c] = 0
    return b

# This sets up the board for gravity to work
# All the neg values in board indicates equal cells which
# should be crushed
def turnNegativesToZeroes(b):
    R = len(b)
    C = len(b[0])
    for r in range(R):
        for c in range(C):
            if b[r][c] < 0:
                b[r][c] = 0
    return b

# Since both hori and vert crushing should be done SIMULTANEOUSLY,
# First mark same cells as their negative val and then call gravity()
# And whenever we are checking for equal cells, use abs() so that we know they are same
def crush(board):
    R = len(board)
    C = len(board[0])
    # Marking same elements as 0 -- in rows
    for r in range(R):
        for c in range(C-2):
            if abs(board[r][c]) == abs(board[r][c+1]) == abs(board[r][c+2]):
                board[r][c] = -board[r][c] if board[r][c] > 0 else board[r][c]
                board[r][c+1] = -board[r][c] if board[r][c] > 0 else board[r][c]
                board[r][c+2] = -board[r][c] if board[r][c] > 0 else board[r][c]

    # Marking same elements as 0 -- in cols
    for r in range(R-2):
        for c in range(C):
            if abs(board[r][c]) == abs(board[r+1][c]) == abs(board[r+2][c]):
                board[r][c] = -board[r][c] if board[r][c] > 0 else board[r][c] # This check is to avoid double negatives
                board[r+1][c] = -board[r][c] if board[r][c] > 0 else board[r][c]
                board[r+2][c] = -board[r][c] if board[r][c] > 0 else board[r][c]

    board = turnNegativesToZeroes(board)

    # Keep dropping the numbers till theres no space to drop anymore
    while shouldGravityBeInvoked(board):
        board = invokeGravity(board)

    return board
            
# board = [[110,5,112,113,114],[210,211,5,213,214],[310,311,3,313,314],[410,411,412,5,414],[5,1,512,3,3],[610,4,1,613,614],[710,1,2,713,714],[810,1,2,1,1],[1,1,2,2,2],[4,1,4,4,1014]]
board = [[1,2,4,5],[3,4,5,6],[3,3,3,7],[1,2,3,5]]
# board = [[1,2,3],[1,4,5],[1,1,1]]
# b = [[0,0,0], [1,2,3], [1,4,5]]
# b2 = [[0,2,3], [0,4,5], [0,0,0]]
# print(candyCrush(board))


# Amazon song selection
def songSelection(arr,k):
    m = []
    for i in range(len(arr)):
        m.append((arr[i], i))

    m.sort(key=lambda x: x[0])

    k -= 30
    left = 0
    right = len(arr) - 1
    pairs = None
    maxi = 0

    while left < right:
        local_minutes = m[left][0] + m[right][0]
        if local_minutes <= k:
            if local_minutes == k :
                return [m[left][1], m[right][1]]
            elif local_minutes > maxi:
                maxi = local_minutes
                pairs = [m[left][1], m[right][1]]
            left += 1
        else:
            right -= 1
    return pairs
    


a = [30,30,40,20]
t = 90
# print(songSelection(a,t))


# a12b56c1a2 ==> a14b56c1
# Add duplicate letter+count pairs together
# Suyash IBM
def betterCompression(s):
    segregrate_inp = re.split("(\d+)", s)
    d = {}
    res = ''
    for i in range(len(segregrate_inp)-1):
        if i % 2 == 0:
            if segregrate_inp[i] not in d:
                d[segregrate_inp[i]] = segregrate_inp[i+1]
            else:
                d[segregrate_inp[i]] = int(d[segregrate_inp[i]]) + int(segregrate_inp[i+1])

    for k, v in sorted(d.items(), key = lambda x:x[0]):
        res += k + str(v)

    print(res)


s = 'a12b56c1a2'
# betterCompression(s)



# Longest palindromic substring
def longestPalindromicSubstring(s):
    DP = [[False for x in range(len(s))] for y in range(len(s))]
    
    left = 0
    right = 0
    for i in range(1, len(s)):
        for j in range(0, i):

            isInnerWordPalin = DP[j+1][i-1] or i-j <= 2 # if s = abcba, inner word is bcb

            if s[i] == s[j] and isInnerWordPalin:
                DP[j][i] = True

                if i - j > right - left: # i-j is len of curr substring
                    right = i
                    left = j

    return s[left:right+1]

s = 'abxx'
# print(longestPalindromicSubstring(s))

# 25. Level order traversal
# This prints each level in a separate array
# BFS
class Node:
    def __init__(self, value):
        self.val = value
        self.left = None
        self.right = None

def levelOrderTraversalPerLevel(root):
    queue = [root]
    levels = []
    level = 0
    if not root:
        return []
    
    while queue:
        levels.append([])
        for i in range(len(queue)):
            root = queue.pop(0)
            levels[level].append(root.val)

            if root.left:
                queue.append(root.left)
            if root.right:
                queue.append(root.right)
        level += 1 
           
    return levels


root = Node(1) 
root.left = Node(2) 
root.right = Node(3) 
root.left.left = Node(4) 
root.left.right = Node(5) 

# print(levelOrderTraversalPerLevel(root))

# Post order traversal
# Left Right Root (but reverse it coz stack)
def postorderTraversal(root):
    out = []
    stack = [root]
    while stack:
        root = stack.pop()
        if root:
            if root.left is not None:
                stack.append(root.left)
            if root.right is not None:
                stack.append(root.right)
            out.append(root.val)
            
    return out[::-1]

# Inorder traversal
# Left Root Right
# Use curr to move left and right
def inorderTraversal(root):
    out = []
    stack = []
    curr = root
    while len(stack) > 0 or curr != None:
        while curr != None:
            stack.append(curr)
            curr = curr.left
            
        curr = stack.pop()
        out.append(curr.val)
        curr = curr.right
    return out

# Preorder traversal
# Root left right..but root right left coz stack
def preorderTraversal(root):
    out = []
    stack = []
    stack.append(root)
    if not root:
        return []

    while stack:
        root = stack.pop()
        out.append(root.val)

        if root.right is not None:
            stack.append(root.right)
                       
        if root.left is not None:
            stack.append(root.left)
    return out

# Depth of a tree
# Push (root, depth) to stack
# increment depth as we go to its children, update max_so_far
def maxDepth(root):
    if not root:
        return 0
    
    stack = [(root,1)] #depth
    max_depth = 0
    while stack:
        if len(stack) > 0:
            root, depth = stack.pop()
            if root.left:
                stack.append((root.left, depth + 1))
            if root.right:
                stack.append((root.right, depth + 1))
            if depth > max_depth:
                max_depth = depth
    return max_depth
      
# Path Sum
# keep decrementing from sum till we reach zero  
def hasPathSum(root, sum):
    if not root:
        return False

    stack = [(root, sum - root.val)]
    while stack:
        node, check = stack.pop()
        if check == 0 and node.left is None and node.right is None:
            return True
        if node.left:
            stack.append((node.left, check - node.left.val))
        if node.right:
            stack.append((node.right, check - node.right.val))
    return False