import numpy as np


def max_min_scaler(data):  # data scaling
    max_value = np.max(data)
    min_value = np.min(data)
    denominator = max_value - min_value
    return(data - min_value)/denominator


def he_initialization(size_prev, size_next):  # for starting initialization
    return np.random.randn(size_prev, size_next) * np.sqrt(2.0/size_prev)
