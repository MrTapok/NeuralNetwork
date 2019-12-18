import numpy as np
from data_work import max_min_scaler
from data_work import data_generation
from data_work import data_to_one_hot
import activation_functions
from layers_work import layer_forward
import math
from learning import NN_running


def run():

    test = [[1,2,-3],
            [-7,5,4]]
    test = np.asarray(test, dtype=float)

    #print(np.sum(test, axis=0))


    div = [1,2,3]
    div2 = [1,4]

    #print(np.dot(test, -1))
    #print(math.e ** (np.dot(test, -1)))
    bias = [1,1,1]
    #print(activation_functions.softmax_forward(np.add(test, bias)))
    #print(activation_functions.softmax_forward(test))
    x_data = []
    y_data = []

    #data_train = np.genfromtxt("mnist_train.csv", delimiter=",")
    #data_test = np.genfromtxt("mnist_test.csv", delimiter=",")
    #x_train = data_train[:, 1:len(data_train[0])]
    #y_train = data_train[:, 0]
    #y_train = data_to_one_hot(y_train)
    #x_test = data_test[:, 1:len(data_train[0])]
    #y_test = data_test[:, 0]
    #y_test = data_to_one_hot(y_test)

    #print(y_test)

    #x_train = max_min_scaler(x_train)
    #x_test = max_min_scaler(x_test)

    learning_rate = 0.001
    iteration_number = 1
    batch_size = 20

    # [x, y, z] - x and y for layer size, z - for choosing activation function
    # relu = 1, sigmoid = 2, softmax = 3
    gen_data = data_generation(20, 8)
    list_architecture = [
        [8, 6, 1],
        [6, 4, 2],
        [4, 2, 3]
    ]
    y_data = data_generation(20, 2)

    NN_running(gen_data, y_data, list_architecture, learning_rate, iteration_number, batch_size)


if __name__ == '__main__':
    run()