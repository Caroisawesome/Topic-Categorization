from util import Sparse_CSR
from scipy.sparse import csr_matrix
import pickle
import sys
import numpy as np
import util

num_iterations = 10000
num_classes = 20
num_instances = 12000

def get_accuracy_score(correct_data, classified_data):
    actual = []
    guess = []
    with open(correct_data, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # Read into list of lists
        tmp = list(list(rec) for rec in csv.reader(csvfile, delimiter=','))
        for row in tmp:
            actual.append(row[0])

    with open(classified_data,'r') as csvfile:
        reader = csv.reader(csvfile)
        # Read into list of lists
        tmp = list(list(rec) for rec in csv.reader(csvfile, delimiter=','))
        for row in tmp:
            guess.append(row[1])

    num_correct = 0
    for i in range(1,len(actual)):
        if (actual[i-1] == guess[i]):
            num_correct+=1
    return num_correct/len(actual)

def create_scipy_csr(filename):
    file1 = open(filename, 'rb')
    matrix = pickle.load(file1)
    file1.close()
    matrix_scipy = csr_matrix((matrix.data, matrix.cols, matrix.rows), dtype=np.float64)
    return matrix, matrix_scipy

def probability_values(W, X):
    matrix = W * X.transpose()
    ones = np.ones((num_classes,num_instances))
    ones = ones.tolist()
    ones = csr_matrix(ones)
    mat = matrix.expm1() + ones
    mat = add_row_of_ones(mat)
    return normalize_matrix(mat)

def add_row_of_ones(matrix):
    lil_mat = matrix.toarray()
    (r,c) = lil_mat.shape
    lil_mat[r-1, :] = 1
    return csr_matrix(lil_mat, dtype=np.float64)

def normalize_matrix(matrix):
    counts = {}

    num_entries = len(matrix.data)
    for i in range(0, num_entries):

        val = matrix.data[i]
        col = matrix.indices[i]
        if (col in counts):
            counts[col] += val
        else:
            counts[col] = val

    for i in range(0, num_entries):
        if (counts[matrix.indices[i]] > abs(1e-10)):
            div_val = matrix.data[i] / counts[matrix.indices[i]]
            matrix.data[i] = div_val
        else:
            matrix.data[i] = 0

    return matrix


def build_delta_matrix(matrix):
    data = []
    row  = []
    col  = []
    for i in range(1, len(matrix.rows)):
        classification = matrix.last_col_value(i)
        data.append(1)
        row.append(classification - 1)
        col.append(i-1)
    delta = csr_matrix((data, (row, col)), dtype=np.float64)
    return delta


def logistic_regression(W, X, Del, eta, lam):
    W1 = W
    for i in range(0, num_iterations):
        WX = probability_values(W1, X)
        W1 = W1 + eta * ((Del - WX) * X - (lam * W1))
    return W1


def classify(matrix):

    counter = 12001
    data = []
    num_rows = len(matrix)
    for i in range(0, num_rows):
        idx = np.argmax(matrix[i])
        data.append([counter, idx+1])
        counter += 1
    util.write_csv('lr_output', data)


if (__name__ == '__main__'):

    if len(sys.argv) < 3:
        print('Must enter commandline arguments <Eta> <Lambda>')
        print("Eta:    0.01 to 0.001")
        print("Lambda: 0.01 to 0.001")
        exit(0)

    eta = float(sys.argv[1])
    lam = float(sys.argv[2])

    mat, matrix = create_scipy_csr('sparse_training_lr')
    test_data, X = create_scipy_csr('sparse_testing_lr')

    obj = np.zeros((num_classes, 61188 + 1))
    obj2 = obj.tolist()
    W = csr_matrix(obj2, dtype=np.float64)
    delta = build_delta_matrix(mat)

    # remove column with class values from training data
    mat_size = matrix.get_shape()
    matrix.resize((mat_size[0], mat_size[1]-1))
    W = logistic_regression(W, matrix, delta, eta, lam)

    Y = W * X.transpose()
    YT = Y.transpose()
    classify(YT.toarray())
