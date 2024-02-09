from split_dataset import Data, extract_and_safe_data
from utils import import_data, export_data
from multilayer_perceptron import Network
from train import train_network
from predict import predict, show_predictions


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Settings for splitting the database')
    parser.add_argument('--training_percentage', type=int, choices=range(0, 100), default=75,
                        help='Percentage of dataset to be used for training')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to the csv file that contains the data')
    parser.add_argument('--train_file', type=str, default='train_data',
                        help='Name of the file where the training data will be saved')
    parser.add_argument('--predict_file', type=str, default='predict_data',
                        help='Name of the file where the prediction data will be saved')
    parser.add_argument('--hidden_layers', type=int, default=[2, 2], nargs='+',
                        help='Amount of nodes in the hidden layer(s)')
    parser.add_argument('--activation_function', type=str, choices=['relu', 'tanh', 'sigmoid'],
                        help='Activation function to use')
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='Learning rate to be used for the neural network')
    parser.add_argument('--amount_of_epochs', type=int, default=20,
                        help='Number of epochs to train the neural network')
    parser.add_argument('--save_file', type=str, default='network',
                        help='Name of the file to save the trained model')
    args = parser.parse_args()

    # Split the dataset into a training and predictions set
    train_data, test_data = Data(args.data_file).split_dataset(args.training_percentage)

    extract_and_safe_data(train_data, args.train_file)
    extract_and_safe_data(test_data, args.predict_file)

    # Generate a neural network and train it with the data.
    dataset = import_data('data/' + args.train_file + '.txt')
    data = dataset['data']
    expected_results = dataset['expected_results']

    network = Network(len(data[0]), args.hidden_layers, args.activation_function)
    train_network(network, data, expected_results, args.learning_rate, args.amount_of_epochs)

    export_data(args.save_file, network.network)

    # Make predictions with the generated neural network
    dataset = import_data('data/' + args.predict_file + '.txt')
    data = dataset['data']
    expected_results = dataset['expected_results']

    network = Network()
    network.network = import_data('data/' + args.save_file + '.txt')
    predictions = predict(network, data)

    show_predictions(predictions, expected_results)
