import numpy as np
from data_work import max_min_scaler
import activation_functions
from layers_work import layer_forward
import math


def run():
    matrix = ([2,5,3],
              [1,-4,-2])
    bias = [1,1]
    a = np.ones(3)
    print(layer_forward(a,matrix,bias,'softmax'))
    #print(activation_functions.relu_forward(a))
    #print(activation_functions.sigmoid(a))
    #data_train = np.genfromtxt("mnist_train.csv", delimiter=",")
    #data_test = np.genfromtxt("mnist_test.csv", delimiter=",")
    #x_train = data_train[:, 1:len(data_train[0])]
    #y_train = data_train[:, 0]
    #x_test = data_test[:, 1:len(data_train[0])]
    #y_test = data_test[:, 0]

    #x_train = max_min_scaler(x_train)
    #x_test = max_min_scaler(x_test)

    #print(len(x_train[0]))


if __name__ == '__main__':
    run()