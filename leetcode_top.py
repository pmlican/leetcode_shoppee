

'''
3[a2[c]] 可以想象竖着摆放，就是一个栈
]
]
c
[
2
a
[
3

3[a2[c]]  如果为数字，计算multi = multi * 10 + int(x)
^
stack = []
res = ""
multi = 3

3[a2[c]]  入栈，重置res和multi, 上次的res和multi入栈
 ^
stack = [3, ""]
res = ""
multi = 0

3[a2[c]]  如果x为字母，拼接到res
  ^
res = "a"
multi = 0

3[a2[c]] 如果x为],出栈取出栈顶(2, "a"), res = last_res + cur_multi * res
      ^
stack = (3, "")
res = "acc"
multi = 0

'''

import random


def decodeString(s):
    stack, res , multi = [], "", 0
    for c in s:
        if c == '[':
            stack.append((multi, res))
            res, multi = "", 0
        elif c == ']':
            cur_multi, last_res = stack.pop()
            res = last_res + cur_multi * res
        elif '0' <= c <= '9':
            multi = multi * 10 + int(c)
        else:
            res += c
    return res

def decodeString1(s):
    def dfs(s, i):
        res, multi = "", 0
        while i < len(s):
            if '0' <= s[i] <= '9':
                multi = multi * 10 + int(s[i])
            elif s[i] == '[':
                i, tmp = dfs(s, i + 1)
                res += multi * tmp
                multi = 0
            elif s[i] == ']':
                return i, res
            else:
                res += s[i]
            i += 1
        return res
    return dfs(s, 0)

# print(decodeString('3[a2[c]]'))


def findKthLargest(nums, k):
    def quickSelect(l, r):
        p = random.randint(l, r)
        nums[p], nums[r] = nums[r], nums[p]

        x, i = nums[r], l
        for j in range(l, r):
            if (nums[j] >= x):
                nums[j], nums[i] = nums[i], nums[j]
                i += 1
        nums[r], nums[i] = nums[i], nums[r]
        if (i == k - 1):
            return nums[i]
        else:
            if i < k - 1:
                return quickSelect(i + 1, r)
            else:
                return quickSelect(l, i - 1)
    return quickSelect(0, len(nums) - 1)

nums = [3,2,1,5,6,4]
# print(findKthLargest(nums, 2))

for j in range(0, 5):
    print(j)
