import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random

# collumns: 4, 8, 9, 11, 14, 15, 16, 18, 19 probably least useful for prediction


class Perceptron():
    def __init__(self, inputs, weights, bias) -> None:
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
        self.weighted_sum = 0
    
    def calculate_weighted_sum(self):
        sum = 0
        for i in range(len(self.inputs)):
            sum += (self.inputs[i] * self.weights[i])
        self.weighted_sum = sum + self.bias
    
    def activate(self):
        self.calculate_weighted_sum()
        output = math.tanh(self.weighted_sum)
        return output


class Layer():
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


class Data():
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


def get_random_weights(input_nodes, output_nodes):
    weights = []
    for i in range(output_nodes):
        temp = []
        for j in range(input_nodes):
            temp.append(float(random.randrange(-100, 100)) / 100)
        weights.append(temp)
    return weights

if __name__ == "__main__":
    data_class = Data("data.csv")
    data = data_class.data

    hidden_layers = 20

    data = data.drop(columns=['diagnosis', 'id'])

    weights = []

    for i in range(len(data.columns)):
        temp = []
        for j in range(len(data.columns)):
            temp.append(0)
        weights.append(temp)

    layer = Layer(data.iloc[0].to_list(), weights, 0.3)
    output = layer.calculate_output()

    for i in range(hidden_layers):
        output_nodes = len(output)
        weights = get_random_weights(output_nodes, int(output_nodes * 0.97))
        layer = Layer(output, weights, 0.2)
        output = layer.calculate_output()

    weights = get_random_weights(30, 2)

    layer = Layer(output, weights, 0.3)
    layer.calculate_output()
    output = layer.soft_max()

    print("M: ", round(output[0] * 100, 2), "%")
    print("D: ", round(output[1] * 100, 2), "%")
