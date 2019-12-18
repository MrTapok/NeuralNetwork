import numpy as np
from layers_work import layer_forward
from layers_work import layer_backward
from data_work import he_initialization


def NN_running(x_data, y_data, layers_list, learning_rate, iteration_number, batch_size):

    w_matrices = []
    biases = []
    activation_functions = []

    for i in range(0, len(layers_list)):
        w_matrices.append(he_initialization(layers_list[i][1], layers_list[i][0]))  # starting initialization using He initialization
        biases.append(np.ones(layers_list[i][1]))  # starting biases are zeroes
        activation_functions.append(layers_list[i][2])  # activation functions sequence

    for i in range(0, iteration_number):
        NN_step(x_data, y_data, w_matrices, biases, activation_functions, learning_rate, batch_size)


def NN_step(x_data, y_data, w_matrices, biases, activation_functions, learning_rate, batch_size):
    number_of_batches = len(x_data)//batch_size

    new_w_matrices = w_matrices.copy()
    new_biases = biases.copy()

    print(new_w_matrices[len(new_w_matrices) - 1])
    print(new_biases[len(new_w_matrices) - 1])

    print(" ---- ")

    h = []
    z = []
    deltas = []

    for i in range(0, number_of_batches):  # batch forward

        working_length = len(w_matrices)

        for j in range(0, working_length):
            if j == 0:
                z_temp, h_temp = layer_forward(x_data[i*batch_size: (i+1)*batch_size], new_w_matrices[j],
                                               new_biases[j], activation_functions[j])
                h.append(h_temp)
                z.append(z_temp)
                deltas.append([])
            else:
                z_temp, h_temp = layer_forward(h[j-1], new_w_matrices[j],
                                               new_biases[j], activation_functions[j])
                h.append(h_temp)
                z.append(z_temp)
                deltas.append([])

        delta_out = (h[working_length - 1] - y_data[i*batch_size: (i+1)*batch_size]) * layer_backward(z[working_length - 1], activation_functions[working_length - 1])
        deltas[working_length-1] = delta_out.copy()

        #print(delta_out)
        #print(working_length)

        for j in range(working_length - 2, -1, -1):
            print(j)
            deltas[j] = np.transpose(np.dot(np.transpose(new_w_matrices[j+1]), np.transpose(deltas[j+1]))) * layer_backward(z[j], activation_functions[j])

        #print(deltas)

        for j in range(0, working_length):
            if j == 0:
                new_w_matrices[j] = new_w_matrices[j] - (learning_rate/batch_size) * (
                    np.dot(np.transpose(deltas[j]), x_data[i*batch_size: (i+1)*batch_size]))
            else:
                new_w_matrices[j] = new_w_matrices[j] - (learning_rate / batch_size) * (
                    np.dot(np.transpose(deltas[j]), h[j - 1]))

            new_biases[j] = new_biases[j] - (learning_rate / batch_size) * np.sum(deltas[j], axis=0)

    print(new_w_matrices[len(new_w_matrices) - 1])
    print(new_biases[len(new_w_matrices) - 1])

    print(" ---- ")


    return new_w_matrices, new_biases



