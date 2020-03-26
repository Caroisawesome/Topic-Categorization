from util import Sparse_CSR
from scipy.sparse import csr_matrix
import scipy
import pickle
import sys
import numpy as np
import util

num_iterations = 10000
num_classes = 20
num_instances = 12000

def create_scipy_csr(filename):
    file1 = open(filename, 'rb')
    matrix = pickle.load(file1)
    file1.close()
    matrix_scipy = csr_matrix((matrix.data, matrix.cols, matrix.rows), dtype=np.float64)
    return matrix, matrix_scipy

def probability_values(W, X):
    matrix = W * X.transpose()
    #matrix = normalize_matrix(matrix)
#    print(matrix.toarray())
    #(w,h) = matrix.get_shape()
#    print(w,h)
    #ones = np.ones((w,h))
    #ones = ones.tolist()
    #ones = csr_matrix(ones, dtype=np.float64)

    #if(len(matrix.data) > 0):
    #   print("before exp",matrix.data[0])
    #mat = matrix.expm1() + ones
    for i in range(0,len(matrix.data)):
        matrix.data[i] = np.exp(matrix.data[i])
    #mat = scipy.exp(matrix)
    #if(len(matrix.data) > 0):
    #    print("after exp",matrix.data[0])
    mat = add_row_of_ones(matrix)
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
        if (abs(counts[matrix.indices[i]]) > 1e-10):
            div_val = matrix.data[i] / counts[matrix.indices[i]]
            matrix.data[i] = div_val
        else:
            matrix.data[i] = 0

    return matrix

def row_normalize_matrix(matrix):
    num_rows = len(matrix.indptr)-1

    for i in range(0, num_rows):
        start_idx = matrix.indptr[i]
        if (i == num_rows-1):
            end_idx = len(matrix.data)
        else:
            end_idx = matrix.indptr[i+1]
        sum = 0
        for j in range(start_idx, end_idx):
            val = matrix.data[j]
            sum += val
        for j in range(start_idx, end_idx):
            if (abs(sum) > 1e-10):
                matrix.data[j] = matrix.data[j] / sum
            else:
                matrix.data[j] = 0
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


def classify(Y):
    sig = 1/(1+np.exp(-Y))
    print(sig.shape)
    idxs = np.argmax(sig, axis=1)
    print(idxs)
    print(len(idxs))
#    print(idxs[0,0])
    counter = 12001
    data = []
    num_rows = len(idxs)
    for i in range(0, num_rows):
        data.append([counter, idxs[i]+1])
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
    #print(obj2)
    #W  = csr_matrix((num_classes, 61188+1), dtype=np.float64)
    W = csr_matrix(obj2, dtype=np.float64)
    delta = build_delta_matrix(mat)

    # remove column with class values from training data
    mat_size = matrix.get_shape()
    matrix.resize((mat_size[0], mat_size[1]-1))
    mat_norm = row_normalize_matrix(matrix)
    W = logistic_regression(W, mat_norm, delta, eta, lam)
    print(W.toarray())

    print(X.get_shape())
    print(W.get_shape())
    #sig = probability_values(W,X)
    #print(sig)
    X = row_normalize_matrix(X)
    Y = W * X.transpose()
    classify(Y.transpose().toarray())
    #get_accuracy_score('lr_output')
