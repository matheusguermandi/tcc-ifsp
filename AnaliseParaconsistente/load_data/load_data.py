from numpy import genfromtxt


def load_csv(path):
    return genfromtxt(path, delimiter=',')


def load_txt(path, separator):
    return genfromtxt(path, delimiter=separator)
