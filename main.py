import pickle
import numpy as np
from util import Sparse_CSR


def create_conditional_totals_matrix(crs_matrix):
    M = np.zeros((20,61189))
    class_totals = np.zeros(20)
    for row in range(0, crs_matrix.num_rows):
        row_start = crs_matrix.rows[row]
        row_end = crs_matrix.get_idx_last_item_in_row(row)
        class_val = crs_matrix.data[row_end] - 1

        for i in range(row_start, row_end):
            col = crs_matrix.cols[i]
            data_val = crs_matrix.data[i]
            M[class_val][col] += data_val
            class_totals[class_val]+=data_val

    return (M, class_totals)

def create_conditional_probabilities_matrix(regular_matrix, class_totals):
    conditional_m = np.zeros((20,61189))
    class_probabilities = np.zeros(20)
    total = 0
    num_rows = len(regular_matrix)

    for i in range(0,num_rows):
        total += class_totals[i]
        for j in range(0,61188):
            if (class_totals[i] > 0):
                if (regular_matrix[i][j] > class_totals[i]):
                    print(regular_matrix[i][j], class_totals[i])
                conditional_m[i][j] = regular_matrix[i][j]/class_totals[i]

    for i in range(0,num_rows):
        class_probabilities[i] = class_totals[i]/total

    return (conditional_m, class_probabilities)

## TODO! not working yet
def convert_matrix_to_CRS(matrix):
    data = []
    cols = []
    rows = []
    idx_data = 0
    flag = False
    for row in matrix:

        for i in range(0, len(row)):
            val = int(row[i])
            if val > 0:
                data.append(val)
                cols.append(i)
                if not flag:
                    idx_data = len(data) - 1
                    flag = True
                    rows.append(idx_data)
        flag = False
    print("data", data)
    matrix = Sparse_CSR(data, rows, cols)
    return matrix

def get_class_word_probabilities(crs_matrix):
    (conditional_totals, class_totals) = create_conditional_totals_matrix(crs_matrix)
    (conditional_probabilities, class_probabilities) = create_conditional_probabilities_matrix(conditional_totals, class_totals)
    return (conditional_probabilities, class_probabilities)

if (__name__ == '__main__'):
    file = open('sparse_testing', 'rb')
    matrix = pickle.load(file)
    file.close()

    (conditional_probability_matrix, class_probabilities) = get_class_word_probabilities(matrix)
    
    print("conditional probability matrix", conditional_probability_matrix)
    print("class probabilities", class_probabilities)
