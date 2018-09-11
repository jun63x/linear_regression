import numpy as np


def min_max_normalization(x):
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm


# データセットのクラス
class Data:
    def __init__(self, data_file_path='auto-mpg.data'):
        with open('auto-mpg.data', 'r') as f:
            self.X = np.asarray(
                [line.split()[:7] for line in f if "?" not in line.split()]
                ).astype(float).T
        self.min_value = np.amin(self.X[-1])
        self.max_value = np.amax(self.X[-1])

    def set_min_max_normalized_data(self):
        self.X = np.asarray([min_max_normalization(row) for row in self.X]).T

    # データをtrain用とtest用に分割
    def set_splited_data(self, n=100):
        np.random.shuffle(self.X)
        self.X_train = self.X[n:]
        self.X_test = self.X[:n]

    def get_data(self):
        self.set_min_max_normalized_data()
        self.set_splited_data()
        X_train = np.delete(self.X_train, 0, axis=1)
        t_train = self.X_train[:, 0]
        X_test = np.delete(self.X_test, 0, axis=1)
        t_test = self.X_test[:, 0]
        return X_train, t_train, X_test, t_test
