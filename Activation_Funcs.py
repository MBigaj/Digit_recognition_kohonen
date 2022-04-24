import numpy as np


def sigmoid(input):
    return 1 / (1 + (np.exp(-input)))


def softmax(input):
    return np.exp(input) / np.sum(np.exp(input), axis=0)


def mse(prediction, truth):
    return np.mean(np.power(truth - prediction, 2))


def mse_deriv(prediction, truth):
    return 2 * (prediction - truth) / truth.size


def tanh(input):
    return np.tanh(input)


def tanh_deriv(input):
    return 1 - np.tanh(input)**2


def relu(input):
    input = input.copy()
    input[input < 0] = 0
    return input


def relu_deriv(input):
    input[input <= 0] = 0
    input[input > 0] = 1
    return input