import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import DictionaryLearning
from mpl_toolkits.mplot3d import Axes3D

iris = load_iris()
iris_data = iris.data
iris_target = iris.target

dl = DictionaryLearning(3)
# iris_data[::2]去除所有偶数行的值
transformed = dl.fit_transform(iris_data[::2])

colors = np.array(list('rgb'))
f = plt.figure()
ax = f.add_subplot(111, projection='3d')
ax.set_title("Traning Set")
ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], color=colors[iris.target[::2]])
plt.show()