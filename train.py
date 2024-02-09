from multilayer_perceptron import Network
from utils import import_data, export_data, create_plot


# Train a network for a fixed number of epochs
def train_network(network, data, expected_results, l_rate, n_epoch):
    n = len(data)
    total_error = 0
    epochs = []
    losses = []
    accuracy = []

    for epoch in range(n_epoch):
        predicted_correctly = 0
        for index, row in enumerate(data):
            outputs = network.forward_propagate(row)
            expected = expected_results[index]

            if expected == 'M':
                expected = [1, 0]
            else:
                expected = [0, 1]

            if (outputs[0] > outputs[1] and expected[0] > expected[1]) or (outputs[0] < outputs[1] and expected[0] < expected[1]):
                predicted_correctly += 1

            square_error = 0
            for i in range(len(expected)):
                error = pow(outputs[i] - expected[i], 2)
                square_error = (square_error + (0.05 * error))
                total_error = total_error + square_error

            network.backward_propagate_error(expected)
            network.update_weights(row, l_rate)
        accuracy.append(round((predicted_correctly / len(data)) * 100, 2))
        total_error = total_error / n
        print('epoch = %d, learning rate = %.3f, error = %.3f' % (epoch + 1, l_rate, round(total_error, 6)))
        epochs.append(epoch + 1)
        losses.append(total_error)

    create_plot(epochs, losses, 'epoch', 'loss', 'Losses')
    create_plot(epochs, accuracy, 'epoch', 'accuracy', 'Accuracy')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Settings for building and training the neural network')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to the file containing the dataset, this should have been produced \
                        by split_dataset.py')
    parser.add_argument('--hidden_layers', type=list, default=[2, 2],
                        help='List of amount of nodes in the hidden layer(s). Length of list is amount of \
                        hidden layers, each value is amount of nodes in the hidden layer')
    parser.add_argument('--activation_function', type=str, choices=['relu', 'tanh', 'sigmoid'],
                        help='Activation function to use')
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='Learning rate to be used for the neural network')
    parser.add_argument('--amount_of_epochs', type=int, default=20,
                        help='Number of epochs to train the neural network')
    parser.add_argument('--save_file', type=str, default='network',
                        help='Name of the file to save the trained model')
    args = parser.parse_args()

    dataset = import_data(args.dataset)
    data = dataset['data']
    expected_results = dataset['expected_results']

    network = Network(len(data[0]), args.hidden_layers, args.activation_function)
    train_network(network, data, expected_results, args.learning_rate, args.amount_of_epochs)

    export_data(args.save_file, network.network)
