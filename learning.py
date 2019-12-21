import numpy as np
from layers_work import layer_forward
from layers_work import layer_backward
from data_work import he_initialization
from data_work import calculate_prediction


def NN_running(x_data, y_data, x_test, y_test, layers_list, learning_rate, iteration_number, batch_size):

    w_matrices = []
    biases = []
    activation_functions = []
    dropout_flags = []
    dropout_probabilities = []

    for i in range(0, len(layers_list)):  # unpacking architecture
        w_matrices.append(he_initialization(layers_list[i][1], layers_list[i][0]))  # starting initialization using He initialization
        biases.append(np.zeros(layers_list[i][1]))  # starting biases are zeroes
        activation_functions.append(layers_list[i][2])  # activation functions sequence
        dropout_flags.append(layers_list[i][3])
        if layers_list[i][3] == 1:
            dropout_probabilities.append(layers_list[i][4])
        else:
            dropout_probabilities.append(0)
    #print(activation_functions)
    accuracy = []

    for i in range(0, iteration_number):
        w_matrices, biases = NN_step(x_data, y_data, w_matrices, biases, activation_functions, learning_rate, batch_size, dropout_flags, dropout_probabilities)
        print(i)
        accuracy.append(calculate_prediction(x_test, y_test, w_matrices, biases, activation_functions))

    return accuracy


def NN_step(x_data, y_data, w_matrices, biases, activation_functions, learning_rate, batch_size, dropout_flags, dropout_probabilities):
    number_of_batches = len(x_data)//batch_size
    new_w_matrices = w_matrices.copy()
    new_biases = biases.copy()
    working_length = len(w_matrices)

    h = []
    z = []
    deltas = []
    masks = []

    for i in range(0, number_of_batches):  # working on all batches

        for j in range(0, working_length):  # creating dropout masks each batch
            if dropout_flags[j] == 1:
                mask = np.random.binomial(1, 1 - dropout_probabilities[j], len(w_matrices[j])) / (1.0 - dropout_probabilities[j])
                masks.append(mask)
            else:
                masks.append([])

        for j in range(0, working_length):  # forward one batch
            if j == 0:
                z_temp, h_temp = layer_forward(x_data[i*batch_size: (i+1)*batch_size], new_w_matrices[j], new_biases[j], activation_functions[j])
            else:
                if dropout_flags[j-1] == 0:
                    z_temp, h_temp = layer_forward(h[j - 1], new_w_matrices[j], new_biases[j], activation_functions[j])
                else:
                    z_temp, h_temp = layer_forward(np.multiply(h[j - 1], masks[j - 1]), new_w_matrices[j], new_biases[j], activation_functions[j])
            h.append(h_temp)
            z.append(z_temp)
            deltas.append([])

        delta_out = (h[working_length - 1] - y_data[i*batch_size: (i+1)*batch_size]) * layer_backward(z[working_length - 1], activation_functions[working_length - 1])
        deltas[working_length-1] = delta_out.copy()  # calculating delta on last layer

        for j in range(working_length - 2, -1, -1):  # calculating deltas on hidden layers
            if dropout_flags[j] == 0:
                deltas[j] = np.dot(deltas[j+1], new_w_matrices[j+1]) * layer_backward(z[j], activation_functions[j])
            else:
                deltas[j] = np.dot(deltas[j+1], new_w_matrices[j+1]) * np.multiply(layer_backward(z[j], activation_functions[j]), masks[j])

        for j in range(0, working_length):  # backpropogation
            if j == 0:
                new_w_matrices[j] = new_w_matrices[j] - (learning_rate/batch_size) * (
                    np.dot(np.transpose(deltas[j]), x_data[i*batch_size: (i+1)*batch_size]))
            else:
                new_w_matrices[j] = new_w_matrices[j] - (learning_rate/batch_size) * (
                    np.dot(np.transpose(deltas[j]), h[j - 1]))

            new_biases[j] = new_biases[j] -\
                        (learning_rate / batch_size) * np.sum(deltas[j], axis=0)

        masks = []
        h = []
        z = []
        deltas = []

    #print(new_biases)
    return new_w_matrices, new_biases



