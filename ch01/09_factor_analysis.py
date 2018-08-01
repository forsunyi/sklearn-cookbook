# 因子分析

# 因子分析与前面介绍的PCA类似。但两者有一个不同之处。PCA是通过对数据进行线性变换获取一个能够解释数据变量的主成分向量空间，这个空间中的每个主成分
# 向量都是正交的。你可以把PCA看成是 N 维数据集降维成 M 维，其中 M<N 。

# 而因子分析的基本假设是，有 M 个重要特征和它们的线性组合（加噪声），能够构成原始的 N 维数据集。也就是说，你不需要指定结果变量（就是最终生成 N 维）
# 而是要指定数据模型的因子数量（ M 个因子）。
from sklearn import datasets
iris = datasets.load_iris()
from sklearn.decomposition import FactorAnalysis
fa = FactorAnalysis(n_components=2)
iris_two_dim = fa.fit_transform(iris.data)
print(iris_two_dim[:5])

from matplotlib import pyplot as plt
f = plt.figure(figsize=(5,5))
ax = f.add_subplot(111)
ax.scatter(iris_two_dim[:, 0], iris_two_dim[:, 1], c=iris.target)
ax.set_title("Factor Analysis 2 Components")
plt.show()