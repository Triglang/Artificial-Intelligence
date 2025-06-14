def newline():
    print("#" * 20)

print("Hello world")
newline()
# 创建一个列表
print("创建一个列表")
l = [1,2,3,4]
print(l)
newline()

# 体验 range
print("体验 range")
r = range(1,19)
print(r)
print(type(r))
newline()

# 创建一个字典
d = {1:"one"}
print(d)
print(d[1])
print(type(d))
newline()

# 尝试用列表作为字典的键
print("尝试用列表作为字典的键")
# d[l] = 1
# print(d)
newline()

# 创建一个元组，尝试用元组作为字典的键
c = (1,2,3)
d[c] = "Cat"
print(c)
print(d[c])
print(type(d[c]))
import math
print(120 * 1024 * 1024 * 1024 * 0.8)