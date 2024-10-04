class Solution:
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        hashset = set() #/ Creating a new variable, hashset, with an empty set
        for n in nums: #/ Iterating through nums
            if n in hashset: #/ Iterating through empty hashset, adding values to hashset
                return True
            hashset.add(n)
        return False
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        return sorted(s) == sorted(t) #Alphabetically sorted Strings have a bool value of true or false
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        prevMap = {}  # val -> index 
        for i, n in enumerate(nums): #
            diff = target - n
            if diff in prevMap:
                return [prevMap[diff], i]
            prevMap[n] = i
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        stack = [] # pair: (index, height) 
        maxArea = 0

        for i, h in enumerate(heights): #/ Gets index, and index values
            start = i #/ Sets start at first index


            while stack and stack[-1][-1] > h: 
                index, height = stack.pop()
                maxArea = max(maxArea, height * (i - index))
                start = index
            stack.append((start, h))
        for i, h in stack:
            maxArea = max(maxArea, h * (len(heights) - i))
        return maxArea
