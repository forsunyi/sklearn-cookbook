import numpy as np
from sklearn.datasets import load_boston
from sklearn.gaussian_process import GaussianProcess

boston = load_boston()
boston_X = boston.data
boston_Y = boston.target
train_set = np.random.choice([True, False], len(boston_Y), p=[.75, .25])

gp = GaussianProcess()
gp.fit(boston_X[train_set], boston_Y[train_set])


