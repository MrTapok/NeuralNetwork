import numpy as np
from data_work import max_min_scaler
from data_work import data_to_one_hot
from learning import NN_running


def run():

    data_train = np.genfromtxt("mnist_train.csv", delimiter=",")
    data_test = np.genfromtxt("mnist_test.csv", delimiter=",")
    x_train = data_train[:, 1:len(data_train[0])]
    y_train = data_train[:, 0]
    y_train = data_to_one_hot(y_train)
    x_test = data_test[:, 1:len(data_train[0])]
    y_test = data_test[:, 0]
    y_test = data_to_one_hot(y_test)
    x_train = max_min_scaler(x_train)
    x_test = max_min_scaler(x_test)

    print("data ready")

    learning_rate = 0.001
    iteration_number = 10000
    batch_size = 32

    # [x, y, z] - x and y for layer size, z - for choosing activation function
    # relu = 1, sigmoid = 2, softmax = 3

    list_architecture_2 = [[784, 10, 3]]

    list_architecture = [
        [784, 200, 2],
        [200, 100, 1],
        [100, 10, 3],
    ]

    list_architecture_1 = [
        [784, 100, 2],
        [100, 10, 3],
    ]

    print("Testing begins")
    NN_running(x_train, y_train, x_test, y_test, list_architecture, learning_rate, iteration_number, batch_size)

if __name__ == '__main__':
    run()