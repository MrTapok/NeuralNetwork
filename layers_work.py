from activation_functions import relu_forward
from activation_functions import sigmoid_forward
from activation_functions import softmax_forward

from activation_functions import relu_df
from activation_functions import sigmoid_df
from activation_functions import softmax_df

import numpy as np


def layer_forward(inputs, matrix, bias, activation_function):
    raw_out = np.add(np.dot(matrix, inputs.T).T, bias)

    if activation_function == 1:
        return raw_out, relu_forward(raw_out)
    if activation_function == 2:
        return raw_out, sigmoid_forward(raw_out)
    if activation_function == 3:
        return raw_out, softmax_forward(raw_out)


def layer_backward(inputs, activation_function):
    if activation_function == 1:
        return relu_df(inputs)
    if activation_function == 2:
        return sigmoid_df(inputs)
    if activation_function == 3:
        return softmax_df(inputs)
