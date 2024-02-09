import json
import matplotlib.pyplot as plt


def create_plot(x, y, x_label, y_label, title):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(left=1, right=len(x))
    plt.title(title)
    plt.grid(linestyle = '--')
    plt.show()


# Export data to a text file
def export_data(filename, data):
    try:
        file = open('data/' + filename + '.txt', 'x')
    except:
        file = open('data/' + filename + '.txt', 'w')
    json.dump(data, file)


# Import data from a text file
def import_data(filename):
    file = open(filename, 'r')
    data = json.load(file)
    return data
