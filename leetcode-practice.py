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

#print(atoi("-1aa3"))

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
            
        positions [s[i]] = i
    return longestSubstring

    
#print(lengthOfLongestSubstring("bbbbb"))

# Longest palindromic substring
def longestPalindrome(s):
    longest = ""
    for i in range(len(s)):
        temp = expandFromCenter(s, i, i)
        if len(temp) > len(longest):
            longest = temp
            
        temp = expandFromCenter(s, i, i+1)
        if len(temp) > len(longest):
            longest = temp
            
    return longest

def expandFromCenter(s, start, end):
    while start >= 0 and end <= len(s)-1 and s[start] == s[end]:
        start -= 1
        end += 1
    return s[start+1:end]

#print(longestPalindrome("bananas"))


# 3 Sum
def threeSum(arr):
    arr.sort()
    result = []
    r = len(arr)-1
    
    for i in range(len(arr)):
        l = i + 1
        while(l < r):
            s = arr[i] + arr[l] + arr[r]
            if s < 0:
                l += 1
            if s > 0:
                r -= 1
            if s == 0:
                result.append([arr[i], arr[l], arr[r]])
                l += 1
                
    return result

# print(threeSum([-1, 0, 1, 2, -1, -4]))


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
    for i in range(len(a)):
        compliment = target - a[i]
        if compliment in d:
            return d[compliment], i
        d[a[i]] = i 
    return None

# No dict -- but value not index returned
def twoSum1(a, target):
    for i in range(len(a)):
        compliment = target - a[i]
        if compliment in a:
            return compliment, a[i]
        
    return None

a = [2,1,3,11]
#print(twoSum(a,14))

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
22. Flatten singly LL
23. Three Sum
24. Given string, insert spaces after words, given dict of valid words
"""

# 1. Comma formatting -Indian and US
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
#comma(n, 'IN')


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

