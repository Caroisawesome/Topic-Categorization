import logistic_regression as lg
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import csv

def plot_confusion_matrix(actual, guess):

    data =  { 'actual': actual,
              'guess': guess
    }

    df = pd.DataFrame(data, columns=['actual','guess'])
    confusion_matrix = pd.crosstab(df['actual'], df['guess'], rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True)
    plt.show()


def generate_confusion_matrix(answers_file, predictions_file):
    actual = []
    guess = []
    with open(answers_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # Read into list of lists
        tmp = list(list(rec) for rec in csv.reader(csvfile, delimiter=','))
        print(tmp.pop(0))
        for row in tmp:
            actual.append(np.int(row[0]))
    actual.pop()

    with open(predictions_file,'r') as csvfile:
        reader = csv.reader(csvfile)
        # Read into list of lists
        tmp = list(list(rec) for rec in csv.reader(csvfile, delimiter=','))
        tmp.pop(0)
        for row in tmp:
            guess.append(np.int(row[1]))

        print("actual", len(actual))
        print("guess", len(guess))

    plot_confusion_matrix(actual, guess)



if (__name__ == '__main__'):
    generate_confusion_matrix('test_col.csv', 'lr_output.csv')

