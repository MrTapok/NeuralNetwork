import numpy as np


def max_min_scaler(data):  # data scaling
    max_value = np.max(data)
    min_value = np.min(data)
    denominator = max_value - min_value
    return(data - min_value)/denominator


def he_initialization(size_prev, size_next):  # for starting initialization
    return np.random.randn(size_prev, size_next) * np.sqrt(2.0/size_prev)


def data_to_one_hot(data):
    new_data = np.zeros((len(data), 10))
    for i in range(len(data)):
        new_data[i, int(data[i])] = 1
    return new_data


def data_generation(size_1, size_2):
    return np.random.rand(size_1, size_2)

def calculate_hidden_delta(delta, w, z):
    # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
    return np.dot(np.transpose(w), delta) * f_deriv(z_l)
