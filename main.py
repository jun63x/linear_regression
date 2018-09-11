import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from data import Data
from linear_regression import LinearRegression


def min_max_normalization_inverse(x, ma, mi):
    return x * (ma - mi) + mi


if __name__ == '__main__':
    # データをロード
    auto_mpg_data = Data()
    max_mpg = auto_mpg_data.max_value
    min_mpg = auto_mpg_data.min_value
    X_train, t_train, X_test, t_test = auto_mpg_data.get_data()
    # ガウス基底関数モデル
    lr = LinearRegression()
    lr.fit(X_train, t_train)
    gbf_prediction = lr.predict(X_test)
    print(
        'gaussian_basis_function r2_score:',
        r2_score(t_test, gbf_prediction)
        )
    # シグモイド関数モデル
    lr = LinearRegression(func_name='sigmoid')
    lr.fit(X_train, t_train)
    sigmoid_prediction = lr.predict(X_test)
    print('sigmoid r2score:', r2_score(t_test, sigmoid_prediction))
    # 結果をプロット
    plt.scatter(
        min_max_normalization_inverse(t_test, max_mpg, min_mpg),
        min_max_normalization_inverse(gbf_prediction, max_mpg, min_mpg),
        c="cyan",
        label="gbf"
        )
    plt.scatter(
        min_max_normalization_inverse(t_test, max_mpg, min_mpg),
        min_max_normalization_inverse(sigmoid_prediction, max_mpg, min_mpg),
        c="magenta",
        label="sigmoid"
        )
    x = np.arange(68, 86, 1)
    y = x
    plt.plot(x, y, c="black")
    plt.xlabel('test')
    plt.ylabel('prediction')
    plt.legend()
    plt.show()
    plt.close()
