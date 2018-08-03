import numpy as np
from scipy.stats import mode
a = np.array([1, 2, 3])
b = np.array([4, 3, 4])
x = np.vstack((a, b))
np.cov(a, b)
np.cov(a, b, bias=True)
np.cov(a)
np.corrcoef(x)

data = [1, 2, 3]
data = np.array([1, 2, 3])
# 创建一组服从正态分布的定量数据
data = np.random.normal(0, 10, size=10)
# 创建一组服从均匀分布的定性数据
data = np.random.randint(0, 10, size=10)

# 计算均值/期望
np.mean(data)
# 计算中位数
np.median(data)

# 计算众数
mode(data)

# 极差max-min
np.ptp(data)
# 方差
np.var(data)
# 标准差
np.std(data)
# 变异系数
np.mean(data) / np.std(data)

