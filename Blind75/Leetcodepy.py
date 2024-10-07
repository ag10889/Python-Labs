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


            while stack and stack[-1][-1] > h: #/ While stack is not empty, and index and height are greater that current height
                index, height = stack.pop() #Saves the index, and heigh values before its popped
                maxArea = max(maxArea, height * (i - index)) #Calculates the maxArea by said height, and by taking the current index, and subtracinttg it from the popped index. 
                start = index #/ new start 
            stack.append((start, h)) #/ Append said new start, and new height to be compared. 
        for i, h in stack: #/ check left over values inside the array 
            maxArea = max(maxArea, h * (len(heights) - i)) #/ checsk said left over values by computing the maxarea using the new h and the original length subtraced by the new i len. 
        return maxArea
    def isPalindrome(self, s: str) -> bool:
        #Is case insensitive
        
        reverseStr = "" # create a empty string to hold the reverse
        for i in s: #Not sure if this is correct, but my idea is to iterate in reverse order through the original string 
            if i.isalpha() or i.isdigit(): #Asking if the element is a digit or alphabetical
                 reverseStr += i.tolower() #Appending the string to the new string, in lower format
        return (reverseStr == reverseStr[::-1]) #returning the new string, and the reversed i string