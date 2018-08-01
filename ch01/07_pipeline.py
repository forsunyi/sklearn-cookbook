from sklearn import datasets
import numpy as np
from sklearn import preprocessing
from sklearn import pipeline

mat = datasets.make_spd_matrix(10)  # 生成一个随机对称正定矩阵
# print(np.random.binomial(9, .3, (10,10))) # 二项分布，进行9次试验，每次正概率为0.3，将结果构成10*10的矩阵
masking_array = np.random.binomial(1, .1, mat.shape).astype(bool)
mat[masking_array] = np.nan
# print(mat[:4, :4])
# print('#################################################################')

# impute = 归咎于，估算
impute = preprocessing.Imputer()
scaler = preprocessing.StandardScaler() # 标准化数据，保证每个维度的特征数据方差为1，均值为0，使得预测结果不会被某些维度过大的特征值而主导
mat_imputed = impute.fit_transform(mat)
# print(mat_imputed[:4, :4])

mat_imp_and_scaled = scaler.fit_transform(mat_imputed)
# print(mat_imp_and_scaled)


# 使用管线命令执行以上同样的操作
pipe = pipeline.Pipeline([('impute', impute), ('scaler', scaler)])
new_mat = pipe.fit_transform(mat)
# print(new_mat[:4, :4])

print(np.array_equal(new_mat, mat_imp_and_scaled))