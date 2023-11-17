import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


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
        random.shuffle(shuffled_data)

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


class MultilayerPerceptron():
    @staticmethod
    def starting_weights(x, y):
        return [[2 * random.random() -1 for i in range(x)] for j in range(y)]

    @staticmethod
    def soft_max(x):
        exponent = np.exp(x - np.max(x))
        y = exponent / exponent.sum()
        return y
    
    @staticmethod
    def get_expected_results(data):
        expected_results = []
        for expected_result in data.diagnosis:
            if expected_result == 'M':
                expected_results.append([1, 0])
            else:
                expected_results.append([0, 1])
        return expected_results
    
    @staticmethod
    def display_learning_curve(epochs, costs):
        plt.plot(epochs, costs)
        plt.xlabel("Epoch")
        plt.ylabel("Total cost")
        plt.show()
    
    def __init__(self, input_data, hidden_layers, learning_rate) -> None:
        self.expected_results = self.get_expected_results(input_data)
        self.input_data = input_data.drop(columns='diagnosis')
        self.input_data = self.input_data.values.tolist()
        self.input_layer = len(self.input_data[0])
        self.hidden_layers = hidden_layers # List of nodes in each hidden layer
        self.hidden_layers_amount = len(self.hidden_layers)
        self.output_layer = 2
        self.learning_rate = learning_rate
        self.max_epochs = 0
        self.bias_hidden_value = -1
        self.bias_output_value = -1
        # self.activation = (lambda x: x * (x > 0)) #relu
        # self.derivative = (lambda x: 1 * (x > 0))
        self.activation = (lambda x: np.tanh(x)) #tanh
        self.derivative = (lambda x: 1 - x ** 2)
        # self.activation = (lambda x: 1 / (1 + np.exp(-x))) #sigmoid
        # self.derivative = (lambda x: x * (1 - x))

        self.weights_hidden = []
        self.weights_hidden.append(self.starting_weights(self.hidden_layers[0], self.input_layer))
        for i in range(1, self.hidden_layers_amount):
            self.weights_hidden.append(self.starting_weights(self.hidden_layers[i], self.hidden_layers[i - 1]))
        self.weights_output = self.starting_weights(self.output_layer, self.hidden_layers[self.hidden_layers_amount - 1])

        self.biases_hidden = []
        for i in range(self.hidden_layers_amount):
            self.biases_hidden.append(np.array([self.bias_hidden_value for i in range(self.hidden_layers[i])]))
        self.bias_output = np.array([self.bias_output_value for i in range(self.output_layer)])

        self.output_hidden = []
        for i in range(self.hidden_layers_amount):
            self.output_hidden.append(0)
        self.output_l2 = 0
    
    def back_propagation(self, inputs):
        # Calculate error between hidden and output layer
        error_output = self.output - self.output_l2
        delta_output = (-1 * error_output) * self.derivative(self.output_l2)

        # Update weights between hidden and output layer (gradient descent)
        for i in range(self.hidden_layers[self.hidden_layers_amount - 1]):    # Loop through nodes in previous layer
            for j in range(self.output_layer):                          # Loop through nodes in output layer
                self.weights_output[i][j] -= self.learning_rate * (delta_output[j] * self.output_hidden[self.hidden_layers_amount - 1])
                self.bias_output[j] -= self.learning_rate * delta_output[j]
        
        # for i, layer in enumerate(self.hidden_layers):
        #     if i == self.hidden_layers_amount:
        #         break
            


        # # Calculate error between input and hidden layer
        # delta_hidden = np.matmul(self.weights_hidden[0], delta_output) * self.derivative(self.output_hidden[0])

        # # Update weights between input and hidden layer (gradient descent)
        # for i in range(self.output_layer):
        #     for j in range(self.hidden_layers[0]):
        #         self.weights_hidden[0][i][j] -= self.learning_rate * (delta_hidden[j] * inputs[i])
        #         self.biases_hidden[0][j] -= self.learning_rate * delta_hidden[j]
    
    def train(self, max_epochs):
        self.max_epochs = max_epochs
        current_epoch = 1
        total_error = 0
        n = len(self.input_data)
        costs = []
        epochs = []

        while(current_epoch <= self.max_epochs):
            for index, inputs in enumerate(self.input_data):
                self.output = self.expected_results[index]

                # Forward propagation
                # inputs is one row of data
                self.output_hidden[0] = self.activation((np.dot(inputs, self.weights_hidden[0]) + self.biases_hidden[0].T))
                for i in range(1, self.hidden_layers_amount):
                    self.output_hidden[i] = self.activation((np.dot(self.output_hidden[i - 1], self.weights_hidden[i]) + self.biases_hidden[i].T))
                self.output_l2 = self.activation((np.dot(self.output_hidden[self.hidden_layers_amount - 1], self.weights_output) + self.bias_output.T))
                prediction = self.soft_max(self.output_l2)

                #Calculate error (cost)
                square_error = 0
                for i in range(self.output_layer):
                    error = pow(self.output[i] - prediction[i], 2)
                    square_error = (square_error + (0.05 * error))
                    total_error = total_error + square_error
                
                self.back_propagation(inputs)
             
            total_error = total_error / n
            
            if current_epoch % 50 == 0 or current_epoch == 1:
                print("Epoch ", current_epoch, "/", self.max_epochs, "  Total error: ", round(total_error, 4))
                costs.append(total_error)
                epochs.append(current_epoch)

            current_epoch += 1

        self.display_learning_curve(epochs, costs)
        return [self.weights_hidden, self.weights_output], [self.bias_hidden, self.bias_output]
    
    def predict(self, input, weights, biases):
        predictions = []
        expected_results = input.diagnosis
        input = input.drop(columns='diagnosis')
        input = input.values.tolist()

        for row in input:
            output_l1 = self.activation(np.matmul(row, weights[0]) + biases[0].T)
            output_l2 = self.activation(np.matmul(output_l1, weights[1]) + biases[1].T)
            prediction = self.soft_max(output_l2)
            if prediction[0] > prediction[1]:
                predictions.append(["M", prediction[0]])
            else:
                predictions.append(["B", prediction[1]])
        
        return predictions, expected_results


if __name__ == "__main__":
    train_data, test_data = Data("data.csv").split_dataset(90)
    
    train_data = train_data.drop(columns=['id'])
    test_data = test_data.drop(columns=['id'])

    multilayer_perceptron = MultilayerPerceptron(train_data, [20, 10, 4], 0.05)

    weights, biases = multilayer_perceptron.train(400)

    predictions, expected_results = multilayer_perceptron.predict(test_data, weights, biases)

    predicted_correctly = 0
    for i in range(len(predictions)):
        prediction = predictions[i][0]
        probability = round(predictions[i][1] * 100, 2)

        if prediction == expected_results[i]:
            predicted_correctly += 1
            print('\033[37m' + "Predicted: ", prediction, "Probability: ", probability, "Expected result: ", expected_results[i], '\033[37m')
        else:
            print('\033[31m' + "Predicted: ", prediction, "Probability: ", probability, "Expected result: ", expected_results[i], '\033[31m')
    
    print()

    print('\033[37m' + "Predicted ", predicted_correctly, " out of ", len(predictions), " correctly", round((predicted_correctly / len(predictions)) * 100, 2), "%", '\033[37m')
