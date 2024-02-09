import pandas as pd
from random import shuffle
import matplotlib.pyplot as plt
from utils import export_data


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


def extract_and_safe_data(data, filename):
    data = data.drop(columns=['id'])
    expected_results = data.diagnosis
    data = data.drop(columns=['diagnosis'])
    data = data.values.tolist()
    expected_results = expected_results.values.tolist()

    export_data(filename, {'data': data, 'expected_results': expected_results})


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
    args = parser.parse_args()

    train_data, test_data = Data(args.data_file).split_dataset(args.training_percentage)

    extract_and_safe_data(train_data, args.train_file)
    extract_and_safe_data(test_data, args.predict_file)
