import math
import numpy as np


# functions for relu

def relu_df(x):
    if x <= 0:
        return 0
    else:
        return 1


def relu_forward(x):
    z = np.zeros(len(x))
    return np.maximum(z,x)

# functions for sigmoid


def sigmoid(x):
    return 1 / (1 + math.e**(-x))


def sigmoid_df(x):
    return sigmoid(x) * (1 - sigmoid(x))


def sigmoid_forward(x):
    return 1 / (1 + math.e**(-x))

# functions for softmax


def softmax_forward(x):
    x = math.e**x
    return x/(np.sum(x))


def softmax_df(x):
    return

