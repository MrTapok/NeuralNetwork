from activation_functions import relu_forward
from activation_functions import sigmoid_forward
from activation_functions import softmax_forward

from activation_functions import relu_df
from activation_functions import sigmoid_df
from activation_functions import softmax_df

import numpy as np


def layer_forward(inputs, matrix, bias, activation_function=0):
    raw_out = np.add(np.dot(matrix, inputs.T).T, bias)
    af_to_af_f = {
        1: relu_forward(raw_out),
        2: sigmoid_forward(raw_out),
        3: softmax_forward(raw_out),
        0: raw_out  # for testing
    }
    try:
        return raw_out, af_to_af_f[activation_function]
    except KeyError as e:
        raise ValueError('Undefined unit: {}'.format(e.args[0]))


def layer_backward(inputs, activation_function=0):
    af_to_af_f = {
        1: relu_df(inputs),
        2: sigmoid_df(inputs),
        3: softmax_df(inputs),
        0: inputs  # for testing
    }
    try:
        return af_to_af_f[activation_function]
    except KeyError as e:
        raise ValueError('Undefined unit: {}'.format(e.args[0]))