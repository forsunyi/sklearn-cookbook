from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
iris_X = iris.data
# binomial构造一个二项分布的样本
# 也即对这个n个数，分别以p进行确定其值为1（选中该值0），
# 以(1-p)确定其值为0（也即是未选中该值）
masking_array = np.random.binomial(1, .25, iris_X.shape).astype(bool)
iris_X[masking_array] = np.nan
print(iris_X[:5])

from sklearn import preprocessing
# strategy的取值有三种：mean 均值、median 中位数、most_frequent 众数
# default="mean"
impute = preprocessing.Imputer(strategy='most_frequent')
iris_X_prime = impute.fit_transform(iris_X)
print(iris_X_prime[:5])

# 用-1代替缺失值
iris_X[np.isnan(iris_X)] = -1
print(iris_X[:5])
# 填充缺失值
impute = preprocessing.Imputer(missing_values=-1)
iris_X_prime = impute.fit_transform(iris_X)
print()