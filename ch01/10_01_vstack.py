import numpy as np

# stack 函数的使用
a = [[1, 2, 3],
     [4, 5, 6]]
print("列表a如下：")
print(a)
print(type(a))

print("横向构建")
c = np.stack(a, axis=0)
print(c)
print(type(c))

print("纵向构建")
c = np.stack(a, axis=1)
print(c)
print(type(c))

# hstack 函数的使用
a = [1, 2, 3]
b = [4, 5, 6]
c = np.hstack((a, b))
print(c)
print(type(c))
a = [[1], [2], [3]]
b = [[1], [2], [3]]
c = [[1], [2], [3]]
d = [[1], [2], [3]]
print(np.hstack((a, b, c, d)))

# vstack 函数使用
a = [1, 2, 3]
b = [4, 5, 6]
c = np.vstack((a, b))
print(c)
a = [[1], [2], [3]]
b = [[1], [2], [3]]
c = [[1], [2], [3]]
d = [[1], [2], [3]]
print(np.vstack((a, b, c, d)))
