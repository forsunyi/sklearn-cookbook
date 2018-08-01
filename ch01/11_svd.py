from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

iris = load_iris()
iris_data = iris.data
print(iris_data.shape)

svd = TruncatedSVD(2)
iris_transformed = svd.fit_transform(iris_data)
print(iris_transformed.shape)

f = plt.figure(figsize=(5, 5))
ax = f.add_subplot(111)
ax.scatter(iris_transformed[:, 0], iris_transformed[:, 1], c=iris.target)
ax.set_title("Truncated SVD, 2 Components")
plt.show()

from scipy.linalg import svd

D = np.array([[1, 2, 3], [1, 3, 4], [1, 4, 5]])

U, S, V = svd(D, full_matrices=False)
np.diag(S)
np.dot(U.dot(np.diag(S)), V)

tsvd = TruncatedSVD()
D_transformed = tsvd.fit_transform(D)