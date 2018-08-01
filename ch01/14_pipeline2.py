from sklearn.datasets import load_iris
import numpy as np
from sklearn import pipeline, preprocessing, decomposition

iris = load_iris()
iris_data = iris.data

mask = np.random.binomial(1, .25, iris_data.shape).astype(bool)
iris_data[mask] = np.nan

pca = decomposition.PCA()
imputer = preprocessing.Imputer()
pipe = pipeline.Pipeline([('imputer', imputer), ('pca', pca)])
np.set_printoptions(2)
iris_data_transformed = pipe.fit_transform(iris_data)

pipe2 = pipeline.make_pipeline(imputer, pca)
iris_data_transformed2 = pipe2.fit_transform(iris_data)


pipe2.set_params(pca__n_components=2)

