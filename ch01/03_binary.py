from sklearn import datasets
import numpy as np
from sklearn import preprocessing
from scipy.sparse import coo

boston = datasets.load_boston()
# 二元特征——通过样本均值进行划分，大于均值为1，小于均值的为0

# binarize函数调用
new_target = preprocessing.binarize(np.array([boston.target]), threshold=boston.target.mean())
# print(new_target)

# Binarizer类
bin = preprocessing.Binarizer(boston.target.mean())
new_target = bin.fit_transform(np.array([boston.target]))
# print(new_target)

# 稀疏矩阵
spar = coo.coo_matrix(np.random.binomial(1, .25, 100))
# preprocessing.binarize(spar, threshold=-1)
print(spar)