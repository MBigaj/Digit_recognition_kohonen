import numpy as np
import os
from layers.Layers import Dense, Activation
from auxilary.activation_functions import sigmoid, sigmoid_deriv, mse, mse_deriv

class NeuralNetwork:
    def __init__(self, learning_rate):
        # NEEDED FOR LOADING DATA
        # self.fcc_1 = Dense(28**2, 100)
        # self.fcc_2 = Dense(100, 50)
        # self.fcc_3 = Dense(50, 10)
        # self.fcc_1.load_data('1')
        # self.fcc_2.load_data('2')
        # self.fcc_3.load_data('3')

        # NEEDED FOR TRAINING
        self.learning_rate = learning_rate
        self.fcc_1 = Dense(28**2, 100)
        self.fcc_2 = Dense(100, 50)
        self.fcc_3 = Dense(50, 10)
        self.activate_1 = Activation()
        self.activate_2 = Activation()
        self.activate_3 = Activation(sigmoid, sigmoid_deriv)

    def flatten(self, input):
        return input.reshape(1, input.shape[0]**2)

    def layers(self, output):
        output = self.fcc_1.forward_prop(output)
        output = self.activate_1.forward_prop(output)
        output = self.fcc_2.forward_prop(output)
        output = self.activate_2.forward_prop(output)
        output = self.fcc_3.forward_prop(output)
        output = self.activate_3.forward_prop(output)
        return output

    def train(self, dataset, templates, epochs):
        data_size = len(dataset)
        for i in range(epochs):
            error = 0
            for j in range(data_size):
                # FORWARD PROPAGATION
                output = self.layers(dataset[j])

                error += mse(output, templates[j])

                # BACKWARD PROPAGATION WITH WEIGHTS UPDATE
                loss = mse_deriv(output, templates[j])
                loss = self.activate_3.backward_prop(loss)
                loss = self.fcc_3.backward_prop(loss, self.learning_rate)
                loss = self.activate_2.backward_prop(loss)
                loss = self.fcc_2.backward_prop(loss, self.learning_rate)
                loss = self.activate_1.backward_prop(loss)
                self.fcc_1.backward_prop(loss, self.learning_rate)

            error /= data_size
            print(f'Epoch: {i+1}/{epochs} - Error: {error}')

        # self.fcc_1.store_data('1')
        # self.fcc_2.store_data('2')
        # self.fcc_3.store_data('3')

    def predict(self, input):

        input = self.flatten(input)
        output = self.layers([input])
        prediction = 0

        parsed_output = output[0][0]
        maximum = parsed_output[0]

        for index, value in enumerate(parsed_output):
            if value > maximum:
                maximum = value
                prediction = index

        print(parsed_output)

        return f'I think that the number you have drawn is the number - {prediction}'