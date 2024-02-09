import numpy as np
from random import random


class Network:
    # Initialize a network
    def __init__(self, number_of_input_nodes=1, hidden_layers=None, activation_function=None):
        self.network = list()
        self.n_output_nodes = 2

        if hidden_layers is None:
            hidden_layers = [2, 2]

        self.hidden_layer = [{'weights': [random() for i in range(number_of_input_nodes + 1)]} for i in range(hidden_layers[0])]
        self.network.append(self.hidden_layer)
        for i in range(1, len(hidden_layers)):
            self.hidden_layer = [{'weights': [random() for i in range(hidden_layers[i - 1] + 1)]} for j in range(hidden_layers[i])]
            self.network.append(self.hidden_layer)
        self.output_layer = [{'weights': [random() for i in range(hidden_layers[-1] + 1)]} for i in range(self.n_output_nodes)]
        self.network.append(self.output_layer)

        if activation_function == 'relu':
            self.activation = (lambda x: x * (x > 0))
            self.derivative = (lambda x: 1 * (x > 0))
        elif activation_function == 'tanh':
            self.activation = (lambda x: np.tanh(x))
            self.derivative = (lambda x: 1 - x ** 2)
        else:
            self.activation = (lambda x: 1 / (1 + np.exp(-x)))
            self.derivative = (lambda x: x * (1 - x))

    # Calculate neuron activation for an input
    @staticmethod
    def activate(weights, inputs):
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]
        return activation

    @staticmethod
    def soft_max(x):
        exponent = np.exp(x - np.max(x))
        x = exponent / exponent.sum()
        return x

    # Transfer neuron activation
    def transfer(self, x):
        return self.activation(x)

    # Calculate the derivative of a neuron output
    def transfer_derivative(self, x):
        return self.derivative(x)

    # Forward propagate input to a network output
    def forward_propagate(self, row):
        inputs = row
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return self.soft_max(inputs)

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            # hidden layers
            if i != len(self.network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['weight_delta'])
                    errors.append(error)
            # output layer
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(neuron['output'] - expected[j])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['weight_delta'] = errors[j] * self.transfer_derivative(neuron['output'])

    # Update network weights with error
    def update_weights(self, row, l_rate):
        for i in range(len(self.network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] -= l_rate * neuron['weight_delta'] * inputs[j]
                neuron['weights'][-1] -= l_rate * neuron['weight_delta']
