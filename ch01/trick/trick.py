import numpy as np
from numpy import linalg as la

matrix = np.mat([[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
                 [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
                 [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
                 [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
                 [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
                 [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
                 [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
                 [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
                 [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
                 [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
                 [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]])

u, sigma, v = la.svd(matrix)


def sigmaPct(sigma, percentage):
    sigma2 = sigma ** 2  # 对sigma求平方
    sumsgm2 = sum(sigma2)  # 求所有奇异值sigma的平方和
    sumsgm3 = 0  # sumsgm3是前k个奇异值的平方和
    k = 0
    for i in sigma:
        sumsgm3 += i ** 2
        k += 1
        if sumsgm3 >= sumsgm2 * percentage:
            return k
    return k


k = sigmaPct(sigma, 0.80)
sigmaK = np.mat(np.eye(k) * sigma[:k])
xformedItems = matrix.T * u[:, :k] * sigmaK.I
unratedItems = np.nonzero(matrix[1, :].A == 0)[1]

print((1024).to_bytes(2, byteorder='big'))