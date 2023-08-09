#####################################################################
import pandas as pd
import numpy as np
#####################################################################

class Solution(object):
    def pivotIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        right_sum = sum(nums)
        left_sum = 0
        for i, num in enumerate(nums):
            right_sum -= num
            if left_sum == right_sum:
                return i
            left_sum += num
        return -1

    def dominantIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        largest_num = max(nums)
        for i in range(len(nums)):
            if nums[i] == largest_num:
                largest_index = i
            elif 2*nums[i] > largest_num:
                return -1
        return largest_index



if __name__ == '__main__':
    nums = [1,7,3,6,5,6]
    solution_instance = Solution()  # Creating an instance of the Solution class
    print(solution_instance.pivotIndex(nums=nums))