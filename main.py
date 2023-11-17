import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random

# columns: 4, 8, 9, 11, 14, 15, 16, 18, 19 probably least useful for prediction


class Data:
    def __init__(self, data_file) -> None:
        col_names = ['id', 'diagnosis']
        for i in range(0, 30):
            col_names.append("col_" + str(i))
        self.data = pd.read_csv(data_file, names=col_names)

        self.standardize()

    def standardize(self):
        for column_id in self.data.columns[2:]:
            column = self.data.get(column_id)
            self.data[column_id] = (column - column.mean()) / column.std()

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

            x = [1, 2] # M = 1, B = 2
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


class Perceptron:
    def __init__(self, inputs, weights, bias) -> None:
        self.inputs = list(inputs)
        self.weights = list(weights)
        self.bias = bias
        self.weighted_sum = 0

    def calculate_weighted_sum(self):
        temp_sum = 0
        for i in range(len(self.inputs)):
            temp_sum += (self.inputs[i] * self.weights[i])
        self.weighted_sum = temp_sum + self.bias

    def activate(self):
        self.calculate_weighted_sum()
        output = math.tanh(self.weighted_sum)
        return output


class Layer:
    def __init__(self, inputs, weights, bias) -> None:
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
        self.output = []

    def calculate_output(self):
        for i in range(len(self.weights)):
            perceptron = Perceptron(self.inputs, self.weights[i], self.bias)
            self.output.append(perceptron.activate())
        return self.output

    def soft_max(self):
        exponent = np.exp(self.output - np.max(self.output))
        self.output = exponent / exponent.sum()
        return self.output


class Network:
    def __init__(self, data, hidden_layers, weights):
        self.data = data
        self.hidden_layers = hidden_layers
        self.weights = weights

    def run_network(self):
        layer = Layer(self.data, self.weights[0], 0.3)
        output = layer.calculate_output()

        for i in range(self.hidden_layers):
            weights = self.weights[i]
            layer = Layer(output, weights, 0.2)
            output = layer.calculate_output()

        weights = get_random_weights(30, 2)

        layer = Layer(output, weights, 0.3)
        layer.calculate_output()
        output = layer.soft_max()
        return output


class Trainer:
    @staticmethod
    def calculate_cost(network_output, expected_result):
        error = pow(expected_result - network_output, 2)
        cost = error[0] + error[1]
        return cost

    def __init__(self, dataset, hidden_layers, nodes_per_layer, learning_rate):
        expected_results = []
        for expected_result in dataset.diagnosis:
            if expected_result == 'M':
                expected_results.append([1, 0])
            else:
                expected_results.append([0, 1])
        self.expected_results = expected_results
        self.input_data = dataset.drop(columns='diagnosis')
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.output = []
        self.deriv = (lambda x: 1 - x ** 2)

        # weights = list_of_all_weights[list_of_layer_weights[list_of_node_weights]]
        self.weights = []
        for i in range(self.hidden_layers + 1):
            input_nodes = nodes_per_layer
            output_nodes = nodes_per_layer
            if i == 0:
                input_nodes = len(self.input_data.columns)
            elif i == self.hidden_layers:
                output_nodes = 2
            temp_layer = []

            for j in range(output_nodes):
                temp_node = []
                for k in range(input_nodes):
                    temp_node.append(float(random.randrange(-100, 100)) / 100)
                temp_layer.append(temp_node)
            self.weights.append(temp_layer)

    def train(self, max_iterations):
        for i in range(max_iterations):
            self.output = []
            total_cost = self.train_one_epoch()
            # print(total_cost)
            # self.gradient_descent(total_cost)
            self.backpropagation()

    def train_one_epoch(self):
        total_cost = 0
        square_error = 0
        for i in range(len(self.input_data)):
            output, cost = self.run_network(self.input_data.iloc[i], self.expected_results[i])
            square_error = (square_error + (0.05 * cost))
            total_cost = total_cost + square_error
            self.output.append(output)
            # print("M: ", round(output[0] * 100, 2), "%")
            # print("B: ", round(output[1] * 100, 2), "%")
        total_cost = total_cost / i
        return total_cost

    def gradient_descent(self, total_cost):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    # weight_derivative = -(2 / len(self.input_data.columns)) * sum(x * (y - y_predicted))
                    weight_derivative = -(2 / len(self.input_data.columns)) * total_cost
                    self.weights[i][j][k] = self.weights[i][j][k] - (self.learning_rate * weight_derivative)
    
    def backpropagation(self):
        delta_output = []
        error_output = np.array(self.output) - np.array(self.expected_results)
        delta_output = ((-1 * error_output) * self.deriv(np.array(self.expected_results)))
        # print(delta_output)
        # for i in range(len(delta_output)):
        #     delta_output[i] = list(delta_output[i])
        #     if delta_output[i][0] == 0.0:
        #         delta_output[i] = delta_output[i][1]
        #     else:
        #         delta_output[i] = delta_output[i][0]
        #     print(delta_output[i])
        # print(delta_output)
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    output_multiplication = (delta_output[k] * self.output[i])
                    if output_multiplication[0] == 0:
                        self.weights[i][j][k] -= (self.learning_rate * output_multiplication[1])
                    else:
                        self.weights[i][j][k] -= (self.learning_rate * output_multiplication[0])

    def run_network(self, data, expected_result):
        network = Network(data, self.hidden_layers, self.weights)

        network_output = network.run_network()
        cost = self.calculate_cost(network_output, expected_result)
        return network_output, cost


def get_random_weights(input_nodes, output_nodes):
    weights = []
    for i in range(output_nodes):
        temp = []
        for j in range(input_nodes):
            temp.append(float(random.randrange(-100, 100)) / 100)
        weights.append(temp)
    return weights


if __name__ == "__main__":
    input_data = Data("data.csv").data

    input_data = input_data.drop(columns=['id'])

    trainer = Trainer(input_data, 3, 7, 0.1)

    trainer.train(500)
