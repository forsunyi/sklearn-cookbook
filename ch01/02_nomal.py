from sklearn import preprocessing
import numpy as np
from sklearn import datasets

boston = datasets.load_boston()
X, y = boston.data, boston.target

print(X[:, :3])
print('前3列的均值和标准差')
print(X[:, :3].mean(axis=0))  # 求前3列特征的均值
print(X[:, :3].std(axis=0))  # 求前3列特征的标准差
print('##################################################################################')

X_2 = preprocessing.scale(X[:, :3])
print(X_2)
print('标准化后前3列的均值和标准差')
print(X_2.mean(axis=0))
print(X_2.std(axis=0))

print('##################################################################################')

my_scaler = preprocessing.StandardScaler()
my_scaler.fit(X[:, :3])
print(my_scaler.transform(X[:, :3]).mean(axis=0))

print('##################################################################################')

my_minmax_scaler = preprocessing.MinMaxScaler()
my_minmax_scaler.fit(X[:, :3])
print('max')
print(my_minmax_scaler.transform(X[:, :3]).max(axis=0))
print('min')
print(my_minmax_scaler.transform(X[:, :3]).min(axis=0))

print('adjust range -3.14 to 3.14')
my_odd_scalar = preprocessing.MinMaxScaler(feature_range=(-3.14, 3.14))
my_odd_scalar.fit(X[:, :3])
print(my_odd_scalar.transform(X[:, :3]).max(axis=0))
print(my_odd_scalar.transform(X[:, :3]).min(axis=0))

print('##################################################################################')

# 正态化
normalized_X = preprocessing.normalize(X[:, :3])
print(normalized_X)

y = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
print(y)
normalized_y = preprocessing.normalize(y)
print(normalized_y)

# 幂等标准化
my_useless_scaler = preprocessing.StandardScaler(with_mean=False, with_std=False)
transformed_sd = my_useless_scaler.fit_transform(X[:, :3]).std(axis=0)
original_sd = X[:, :3].std(axis=0)
print(np.array_equal(transformed_sd, original_sd))
print('transformed_sd = ', transformed_sd)
print('original_sd = ', original_sd)
