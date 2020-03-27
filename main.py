import pickle
import numpy as np
import math
import util
import sys
from util import Sparse_CSR

num_words = 61189
num_classes = 20

def create_conditional_totals_matrix(crs_matrix):
    M = np.zeros((num_classes, num_words))
    class_totals = np.zeros(num_classes)
    row_sums = np.zeros(num_classes)

    for row_num in range(0, crs_matrix.num_rows):
        row_start = crs_matrix.rows[row_num]
        row_end = crs_matrix.get_idx_last_item_in_row(row_num)
        class_val = (crs_matrix.data[row_end]) - 1

        row_total = sum(crs_matrix.data[row_start:row_end])
        row_sums[class_val] += row_total

        for i in range(row_start, row_end):
            col = crs_matrix.cols[i]
            data_val = crs_matrix.data[i]
            M[class_val][col] += data_val

        class_totals[class_val]+= 1 ## data_val

    return (M, class_totals, row_sums)

def create_conditional_probabilities_matrix(regular_matrix, class_totals, row_sums, alpha):

    conditional_m = np.zeros((num_classes,num_words))
    class_probabilities = np.zeros(num_classes)
    num_rows = len(regular_matrix)
    total = sum(class_totals)

    for i in range(0,num_rows):
        class_probabilities[i] = class_totals[i]/total
        for j in range(0,num_words):
            conditional_m[i][j] = (regular_matrix[i][j]+(alpha - 1))/(row_sums[i]+((alpha-1)*num_words))

    return (conditional_m, class_probabilities)

def get_class_word_probabilities(crs_matrix, alpha):
    (conditional_totals, class_totals, row_sums) = create_conditional_totals_matrix(crs_matrix)
    (conditional_probabilities, class_probabilities) = create_conditional_probabilities_matrix(conditional_totals, class_totals, row_sums, alpha)
    return (conditional_probabilities, class_probabilities)

def classify_row(row_num, class_prob, cond_prob_matrix, testing_csr):
    probabilities = []
    classes       = []
    max_idx       = 0
    row_idx       = testing_csr.rows[row_num]

    for j in range(0, len(class_prob)):
        x = math.log2(class_prob[j])
        likelihood = 0
        for i in range(row_idx, testing_csr.get_idx_last_item_in_row(row_num)):
            col_idx    = testing_csr.cols[i]
            likelihood += testing_csr.data[i] * (math.log2(cond_prob_matrix[j][col_idx]))
        classes.append(j + 1 )
        probabilities.append(x + likelihood)

    idx = np.argmax(probabilities)
    return classes[idx]

def classify(cond_prob_matrix, class_prob, testing_csr):
    data = []
    counter = 12001
    for i in range(0, len(testing_csr.rows)):
        class_id = classify_row(i, class_prob, cond_prob_matrix, testing_csr)
        data.append([counter, class_id])
        counter += 1
    util.write_csv('output_nb', data)

def multi_classification_nb(b):
    #if len(sys.argv) < 2:
    #    print("Must enter commandline arguments <Beta>")
    #    print("Beta: between 0.00001 and 1")
    #    exit(0)

    beta = b
    file = open('sparse_training_nb', 'rb')
    file2 = open('sparse_testing_nb', 'rb')
    matrix = pickle.load(file)
    matrix2 = pickle.load(file2)
    file.close()
    file2.close()
    #beta = 1/61188

    alpha = 1 + beta
    (conditional_probability_matrix, class_probabilities) = get_class_word_probabilities(matrix, alpha)
    classify(conditional_probability_matrix, class_probabilities, matrix2)
    score = util.get_accuracy_score('test_col.csv', 'output_nb.csv')
    return score


if (__name__ == '__main__'):

    if len(sys.argv) < 2:
        print("Must enter commandline arguments <Beta>")
        print("Beta: between 0.00001 and 1")
        exit(0)

    beta = float(sys.argv[1])
    file = open('sparse_training_nb', 'rb')
    file2 = open('sparse_testing_nb', 'rb')
    matrix = pickle.load(file)
    matrix2 = pickle.load(file2)
    file.close()
    file2.close()
    #beta = 1/61188

    alpha = 1 + beta
    (conditional_probability_matrix, class_probabilities) = get_class_word_probabilities(matrix, alpha)
    classify(conditional_probability_matrix, class_probabilities, matrix2)
    score = util.get_accuracy_score('test_col.csv', 'output_nb.csv')
    print(score)
