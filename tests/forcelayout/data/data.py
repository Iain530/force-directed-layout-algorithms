import numpy as np


def load_iris():
    with open('tests/forcelayout/data/iris_small.csv', encoding='utf8') as data_file:
        return np.loadtxt(
            data_file,
            skiprows=1,
            delimiter=',',
            comments='#'
        )
