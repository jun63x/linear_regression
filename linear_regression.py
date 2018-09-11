import numpy as np


# ガウス基底関数
def gaussian_basis_function(x, s=0.1):
    return np.append(1, np.exp(-(x - np.arange(0, 0.6, s)) ** 2/(2 * s * s)))


# シグモイド基底関数
def sigmoid(x):
    mu = [0.2 * (i - 5) if i > 5 else 0.2 * (-i) for i in range(6)]
    s = [0.1 for i in range(6)]
    a = (x - mu) / s
    return np.append(1, 1 / (1 + np.exp(-a)))


# 線形回帰のクラス
class LinearRegression:
    def __init__(self, func_name='gbf'):
        func_dic = {'gbf': gaussian_basis_function, 'sigmoid': sigmoid}
        self.func = func_dic[func_name]

    def fit(self, X, t):
        self.phi = np.asarray([self.func(row) for row in X])
        self.omega_ml = np.dot(
            (np.dot(np.linalg.inv(np.dot(self.phi.T, self.phi)), self.phi.T)),
            t
            )

    def predict(self, X_test):
        phi_test = np.asarray(
            [self.func(row) for row in X_test]
            )
        return np.dot(phi_test, self.omega_ml)
