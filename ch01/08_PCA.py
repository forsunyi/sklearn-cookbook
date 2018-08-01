from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
# print(iris_X[:5])
# 导入分解模块
from sklearn import decomposition

# 初始化PCA对象
pca = decomposition.PCA()
iris_pca = pca.fit_transform(iris_X)
# print(iris_pca[:5])
dot = np.dot(iris_pca.T, iris_pca)
pca_v_1 = iris_pca[:, :1]
pca_v_2 = iris_pca[:, 1:2]
# print(np.dot(pca_v_1.T, pca_v_2))

# 数据降维操作，将iris数据降维为2维
pca = decomposition.PCA(n_components=2)
iris_X_prime = pca.fit_transform(iris_X)
# print(iris_X_prime[:5])

# 降维后的数据作图表示
from matplotlib import pyplot as plt
f = plt.figure(figsize=(5, 5))
ax = f.add_subplot(111)
ax.scatter(iris_X_prime[:, 0], iris_X_prime[:, 1], c=iris_y)
ax.set_title("PCA 2 Components")
plt.show()

# 查看降维后保留多少变量信息
print(pca.explained_variance_ratio_.sum()) # 0.9776317750248034 = 97.76%的信息被保留

# 开始设置解释变量保留比例
pca = decomposition.PCA(n_components=.98)
iris_X_prime = pca.fit_transform(iris_X)
print(pca.explained_variance_ratio_.sum())
print(iris_X_prime.shape)


