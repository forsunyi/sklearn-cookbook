from sklearn import datasets
import numpy as np
from sklearn import preprocessing

iris = datasets.load_iris()
X = iris.data
y = iris.target
d = np.column_stack((X, y))

text_encoder = preprocessing.OneHotEncoder()
# te = text_encoder.fit_transform(d[: -1:]).toarray()
# print(text_encoder.transform(np.ones((3, 1))).toarray())


from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer()
my_dict = [{'species': iris.target_names[i]} for i in y]
dv.fit_transform(my_dict).toarray()
# print(dv.fit_transform(my_dict).toarray())

import patsy

patsy.dmatrix("0 + C(species)", {"species": iris.target})
# print(patsy.dmatrix("0 + C(species)", {"species" : iris.target}))
