from sklearn import datasets
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import probplot

boston = datasets.load_boston()
lr = LinearRegression()
lr.fit(boston.data, boston.target)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

# predictions是线性回归的估计值
predictions = lr.predict(boston.data)
f, ax = plt.subplots(figsize=(7, 5))
f.tight_layout()
ax.hist(boston.target - predictions, bins=40, label='Residuals Linear', color='b', alpha=.5)
ax.set_title("Histogram of Residuals")
ax.legend(loc='best')
plt.show()

np.mean(boston.target - predictions)

f = plt.figure(figsize=(7, 5))
ax = f.add_subplot(111)
probplot(boston.target - predictions, plot=ax)
f.show()


# 均方误差
def MSE(target, predictions):
    squared_deviation = np.power(target - predictions, 2)
    return np.mean(squared_deviation)


mse = MSE(boston.target, predictions)


# 平均绝对误差
def MAD(target, predictions):
    absolute_deviation = np.abs(target - predictions)
    return np.mean(absolute_deviation)


mad = MAD(boston.target, predictions)

n_bootstraps = 1000
len_boston = len(boston.target)
subsample_size = np.int(0.5 * len_boston)
subsample = lambda : np.random.choice(np.arange(0, len_boston), size=subsample_size)
coefs = np.ones(n_bootstraps)
for i in range(n_bootstraps):
    subsample_idx = subsample()
    subsample_X = boston.data[subsample_idx]
    subsample_Y = boston.target[subsample_idx]
    lr.fit(subsample_X, subsample_Y)
    coefs[i] = lr.coef_[0]

f = plt.figure(figsize=(7, 5))
ax = f.add_subplot(111)
ax.hist(coefs, bins=50, color='b', alpha=.5)
ax.set_title("Histogram of the lr.coef_[0]")
f.show()

p = np.percentile(coefs, 0.1)