from math import exp
from random import seed, shuffle, random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Data:
    def __init__(self, data_file) -> None:
        self.col_names = ['id', 'diagnosis']
        for i in range(0, 30):
            self.col_names.append("col_" + str(i))
        self.data = pd.read_csv(data_file, names=self.col_names)

        self.standardize()

    def standardize(self):
        for column_id in self.data.columns[2:]:
            column = self.data.get(column_id)
            self.data[column_id] = (column - column.mean()) / column.std()

    def split_dataset(self, train_percent):
        train_percent = train_percent / 100
        shuffled_data = self.data.values.tolist()
        shuffle(shuffled_data)

        train_list = shuffled_data[:int(len(shuffled_data) * train_percent)]
        test_list = shuffled_data[int(len(shuffled_data) * train_percent):]

        train_data = pd.DataFrame(train_list, columns=self.col_names)
        test_data = pd.DataFrame(test_list, columns=self.col_names)

        return train_data, test_data

    def create_plots(self):
        dataM = self.data[self.data['diagnosis'] == 'M']
        dataB = self.data[self.data['diagnosis'] == 'B']

        fig, ax = plt.subplots(6, 5)

        i = 0
        j = 0

        for column in self.data.columns[2:]:
            if i > 4:
                i = 0
                j += 1

            columnM = dataM.describe().get(column)
            columnB = dataB.describe().get(column)
            subplot = ax[j, i]

            x = [1, 2]  # M = 1, B = 2
            y = [columnM['mean'], columnB['mean']]
            e = [columnM['std'], columnB['std']]

            subplot.errorbar(x, y, e, linestyle='None', marker='^', capsize=3)

            subplot.set_xlim((0, 3))
            subplot.set_xticks([1, 2])
            subplot.set_xticklabels(['M', 'B'])

            subplot.set_title(column)

            i += 1

        plt.xlabel('Diagnosis')
        plt.show()


class Network:
    # Initialize a network
    def __init__(self, number_of_input_nodes, hidden_layers=None, activation_function=None):
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

    @staticmethod
    # Calculate neuron activation for an input
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


def get_expected_results(data):
    expected_results = []
    for expected_result in data.diagnosis:
        if expected_result == 'M':
            expected_results.append([1, 0])
        else:
            expected_results.append([0, 1])
    return expected_results


# Train a network for a fixed number of epochs
def train_network(network, data, expected_results, l_rate, n_epoch):
    n = len(data)
    total_error = 0
    for epoch in range(n_epoch):
        for index, row in enumerate(data):
            outputs = network.forward_propagate(row)
            expected = expected_results[index]

            square_error = 0
            for i in range(len(expected)):
                error = pow(outputs[i] - expected[i], 2)
                square_error = (square_error + (0.05 * error))
                total_error = total_error + square_error

            network.backward_propagate_error(expected)
            network.update_weights(row, l_rate)
        total_error = total_error / n
        print('epoch = %d, learning rate = %.3f, error = %.3f' % (epoch + 1, l_rate, round(total_error, 6)))


# Make predictions with the network
def predict(network, data):
    predictions = []
    for index, row in enumerate(data):
        prediction = network.forward_propagate(row)
        if prediction[0] > prediction[1]:
            predictions.append(["M", prediction[0]])
        else:
            predictions.append(["B", prediction[1]])
    return predictions


# Test neural network
seed(3)
train_data, test_data = Data("data.csv").split_dataset(80)

train_data = train_data.drop(columns=['id'])

expected_results_train = get_expected_results(train_data)
train_data = train_data.drop(columns='diagnosis')
train_data = train_data.values.tolist()

network = Network(len(train_data[0]), [2, 2], 'tanh')
train_network(network, train_data, expected_results_train, 0.05, 20)

test_data = test_data.drop(columns=['id'])
expected_results_test = test_data.diagnosis
test_data = test_data.drop(columns='diagnosis')
test_data = test_data.values.tolist()

predictions = predict(network, test_data)
predicted_correctly = 0
for i in range(len(predictions)):
    prediction = predictions[i][0]
    probability = round(predictions[i][1] * 100, 2)

    if prediction == expected_results_test[i]:
        predicted_correctly += 1
        print('\033[37m' + "Predicted: ", prediction, "Probability: ", probability, "Expected result: ", expected_results_test[i], '\033[37m')
    else:
        print('\033[31m' + "Predicted: ", prediction, "Probability: ", probability, "Expected result: ", expected_results_test[i], '\033[31m')

print()

print('\033[37m' + "Predicted ", predicted_correctly, " out of ", len(predictions), " correctly", round((predicted_correctly / len(predictions)) * 100, 2), "%", '\033[37m')

