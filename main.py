import pickle
import numpy as np
import math
import util
from util import Sparse_CSR


def create_conditional_totals_matrix(crs_matrix):
    M = np.zeros((20,61188))
    class_totals = np.zeros(20)
    for row in range(0, crs_matrix.num_rows):
        row_start = crs_matrix.rows[row]
        row_end = crs_matrix.get_idx_last_item_in_row(row)
        class_val = crs_matrix.data[row_end] - 1

        for i in range(row_start, row_end):
            col = crs_matrix.cols[i] - 1
            data_val = crs_matrix.data[i]
            M[class_val][col] += data_val
            class_totals[class_val]+=data_val

    return (M, class_totals)

def create_conditional_probabilities_matrix(regular_matrix, class_totals, alpha):
    conditional_m = np.zeros((20,61188))
    class_probabilities = np.zeros(20)
    total = 0
    num_rows = len(regular_matrix)

    for i in range(0,num_rows):
        total += class_totals[i]
        for j in range(0,61188):
            if (class_totals[i] > 0):
                if (regular_matrix[i][j] > class_totals[i]):
                    print(regular_matrix[i][j], class_totals[i])
                conditional_m[i][j] = (regular_matrix[i][j]+(alpha -1))/(class_totals[i]+((alpha-1)*61188))

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

def get_class_word_probabilities(crs_matrix, alpha):
    (conditional_totals, class_totals) = create_conditional_totals_matrix(crs_matrix)
    (conditional_probabilities, class_probabilities) = create_conditional_probabilities_matrix(conditional_totals, class_totals, alpha)
    return (conditional_probabilities, class_probabilities)

def classify_row(row_num, class_prob, cond_prob_matrix, testing_csr):
    probabilities = []
    classes       = []
    max_idx       = 0
    row_idx       = testing_csr.rows[row_num]
    for j in range(0, len(class_prob)):
        x = math.log2(class_prob[j])
        for i in range(row_idx, testing_csr.get_idx_last_item_in_row(row_num)):
            col_idx    = testing_csr.cols[i]
            likelihood = testing_csr.data[i] * (math.log2(cond_prob_matrix[j][col_idx]))
            probabilities.append(x + likelihood)
            classes.append(j + 1)
    idx = probabilities.index(max(probabilities))
    return classes[idx]

def classify(cond_prob_matrix, class_prob, testing_csr):
    data = []
    counter = 12001
    for i in range(0, len(testing_csr.rows)):
        class_id = classify_row(i, class_prob, cond_prob_matrix, testing_csr)
        data.append([counter, class_id])
        counter += 1
    util.write_csv('output', data)


if (__name__ == '__main__'):


    if len(sys.argv) < 2:
        print("Must enter commandline arguments <Beta>")
        print("Beta: between 0.00001 and 1")
        exit(0)

    beta = float(sys.argv[1])
    file = open('sparse_training', 'rb')
    file2 = open('sparse_testing', 'rb')
    matrix = pickle.load(file)
    matrix2 = pickle.load(file2)
    file.close()
    file2.close()
    #beta = 1/61188
    alpha = 1 + beta
    (conditional_probability_matrix, class_probabilities) = get_class_word_probabilities(matrix, alpha)
    classify(conditional_probability_matrix, class_probabilities, matrix2)
