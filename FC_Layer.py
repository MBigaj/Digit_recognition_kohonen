import os

import numpy as np

class FC:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_prop(self, input):
        self.input = input
        output = np.dot(input, self.weights) + self.bias
        return output

    def backward_prop(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

    # def store_data(self, name):
    #     with open(f'data/weights_{name}.txt', 'w') as file:
    #         for weight in self.weights:
    #             file.write(np.array2string(weight))
    #
    # def load_data(self, name):
    #     with open(f'data/weights_{name}.txt', 'r') as file:
    #         self.weights = np.loadtxt(file)