from activation_functions import relu_forward
from activation_functions import sigmoid_forward
from activation_functions import softmax_forward
import numpy as np


def layer_forward(inputs, matrix, bias, activation_function):
    raw_out = np.dot(matrix, inputs.T) + bias
    af_to_af_f = {
        'relu': relu_forward(raw_out),
        'sigmoid': sigmoid_forward(raw_out),
        'softmax': softmax_forward(raw_out),
        'raw': raw_out  # for testing
    }
    try:
        return af_to_af_f[activation_function]
    except KeyError as e:
        raise ValueError('Undefined unit: {}'.format(e.args[0]))