import math
import numpy as np


# functions for relu

def relu_df(x):
    return np.where(x <= 0, 0, 1)


def relu_forward(x):  # работает
    return np.where(x > 0, x, 0)

# functions for sigmoid


def sigmoid_df(x):  # работает
    s_f = sigmoid_forward(x)
    return s_f * (1 - s_f)


def sigmoid_forward(x):  # работает
    return 1 / (1 + math.e**(np.dot(x, -1)))

# functions for softmax


def softmax_df(x):
    s_f = softmax_forward(x)
    return s_f * (1 - s_f)


def softmax_forward(x):  # работает
    temp = math.e**x
    div = temp.sum(axis=1)
    return np.transpose((np.divide(np.transpose(temp), np.transpose(div))))




