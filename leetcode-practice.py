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



    






    
