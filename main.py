import numpy as np
from data_work import max_min_scaler
from data_work import data_to_one_hot
from learning import NN_running
from data_work import acc_plot

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

    print("Data is ready")

    learning_rate = 0.001
    iteration_number = 30
    batch_size = 32

    # [x, y, z, d, p] - x and y for layer size, z - for choosing activation function, d - for dropout, p - for dropout probability (if d = 1)
    # relu = 1, sigmoid = 2, softmax = 3
    # 0 - no dropout, 1 - dropout

    list_architecture_2 = [[784, 10, 3, 0]]

    list_architecture = [
        [784, 300, 1, 1, 0.3],
        [300, 100, 1, 0],
        [100, 10, 3, 0],
    ]

    list_architecture_1 = [
        [784, 400, 2, 1, 0.3],
        [400, 10, 3, 0],
    ]

    print("Let the fun begin")
    accuracy = NN_running(x_train, y_train, x_test, y_test, list_architecture_1, learning_rate, iteration_number, batch_size)
    acc_plot(accuracy)

if __name__ == '__main__':
    run()