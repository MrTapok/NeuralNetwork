import numpy as np
from layers_work import layer_forward
from data_work import he_initialization


def NN_running(x_data, y_data, layers_list, learning_rate, iteration_number):
    w_matrices = []
    biases = []
    activation_functions = []
    for i in range(0, len(layers_list)):
        w_matrices.append(he_initialization(layers_list[i][1], layers_list[i][0]))  # starting initialization using He initialization
        biases.append(np.zeros(layers_list[i][1])) # starting biases are zeroes
        activation_functions.append(layers_list[i][2]) #

    for i in range(0, len(layers_list)):
        print(len(w_matrices[i]))
        print(len(biases[i]))
    print(w_matrices)
    print(biases)
    print(activation_functions)
    for i in range(0, iteration_number):
        NN_step(x_data, y_data, w_matrices, biases, activation_functions, learning_rate)


def NN_step(x_data, y_data, w_matrices, biases, activation_functions, learning_rate):
    for i in range(0, len(w_matrices)):
        True



