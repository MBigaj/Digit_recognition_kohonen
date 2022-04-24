import numpy as np

from Activation_Funcs import relu, relu_deriv, tanh, tanh_deriv

class Activation:
    def __init__(self):
        self.activation = tanh
        self.activation_deriv = tanh_deriv

    def forward_prop(self, input):
        self.input = input
        return self.activation(input)

    def backward_prop(self, output_error):
        return self.activation_deriv(self.input) * output_error