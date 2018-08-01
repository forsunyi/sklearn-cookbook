from sklearn import datasets
import sklearn.datasets as d
import numpy as np
from matplotlib import pyplot as plt

# boston = datasets.load_boston()
# print(boston.DESCR)

# housing = datasets.fetch_california_housing()
# print(housing.DESCR)

# X, y = boston.data, boston.target

reg_data = d.make_regression()
# print(reg_data[0].shape,reg_data[1].shape)
# print(reg_data[1])

classification_set = d.make_classification(weights=[0.1])
print(np.bincount(classification_set[1]))

blobs = d.make_blobs(200)

f= plt.figure(figsize=(8, 4))

ax = f.add_subplot(111)
ax.set_title("A blob with 3 centers")

colors = np.array(['r', 'g', 'b'])
ax.scatter(blobs[0][:, 0], blobs[0][:, 1], color=colors[blobs[1].astype(int)], alpha=0.75)

# plt.show()

