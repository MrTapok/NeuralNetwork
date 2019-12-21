import numpy as np
from layers_work import layer_forward
from matplotlib import pyplot as plt


def max_min_scaler(data):  # data scaling
    max_value = np.max(data)
    min_value = np.min(data)
    denominator = max_value - min_value
    return(data - min_value)/denominator


def he_initialization(size_prev, size_next):  # HE initialization for matrix weights
    return np.random.randn(size_prev, size_next) * np.sqrt(2.0/size_prev)


def data_to_one_hot(data):  # one hot encoding
    new_data = np.zeros((len(data), 10))
    for i in range(len(data)):
        new_data[i, int(data[i])] = 1
    return new_data


def calculate_by_argmax(y_data, prediction):  # argmax for MSE gradient
    argmaxes = np.argmax(prediction, axis=1)
    temp = np.choose(argmaxes, y_data.T) - np.ones(len(prediction))
    temp = temp * temp
    return np.sum(temp) / len(y_data)


def calculate_prediction(x_data, y_data, w_matrices, biases, activation_functions):  # prediction for accuracy
    h = []
    for i in range(0, len(w_matrices)):  # forward one batch

        if i == 0:
            z_temp, h_temp = layer_forward(x_data, w_matrices[i],
                                           biases[i], activation_functions[i])
            h = h_temp
        else:
            z_temp, h_temp = layer_forward(h, w_matrices[i],
                                           biases[i], activation_functions[i])
            h = h_temp

    return calculate_by_argmax(y_data, h)


def acc_plot(accuracy):  # plot creation
    plt.figure(figsize=(7, 7))
    plt.plot(accuracy, label="Accuracy")
    plt.legend()
    plt.show()
