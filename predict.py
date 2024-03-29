import math
from multilayer_perceptron import Network
from utils import import_data


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


def show_predictions(predictions, expected_results):
    predicted_correctly = 0
    log_loss = 0
    for i in range(len(predictions)):
        prediction = predictions[i][0]
        probability = round(predictions[i][1] * 100, 2)
        log_loss += math.log(predictions[i][1])

        if prediction == expected_results[i]:
            predicted_correctly += 1
            print('\033[37mPredicted: %s - Probability: %d - Expected result: %s' % (prediction, probability, expected_results[i]))
        else:
            print('\033[31mPredicted: %s - Probability: %d - Expected result: %s' % (prediction, probability, expected_results[i]))

    log_loss *= -(1 / len(predictions))

    print()
    print('\033[37mPredicted %d out of %d correctly %0.1f%%' % (predicted_correctly, len(predictions), round((predicted_correctly / len(predictions)) * 100, 2)))
    print()
    print('Binary cross entropy error: %0.5f' % (round(log_loss, 5)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Settings for building and testing the neural network')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to the file containing the dataset, this should have been produced \
                            by split_dataset.py')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the file containing the model, this should have been produced \
                             by train.py')
    args = parser.parse_args()

    dataset = import_data(args.dataset)
    data = dataset['data']
    expected_results = dataset['expected_results']

    network = Network()
    network.network = import_data(args.model)
    predictions = predict(network, data)

    show_predictions(predictions, expected_results)
