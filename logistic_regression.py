from util import Sparse_CSR
from scipy.sparse import csr_matrix
from sklearn.preprocessing import *
import scipy
import pickle
import sys
import numpy as np
import util

num_iterations = 1000
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
    #matrix = matrix.expm1()
    for i in range(0,len(matrix.data)):
        matrix.data[i] = np.exp(matrix.data[i])
    mat = add_row_of_ones(matrix)
    return normalize(mat, norm='l1',axis=0)

def add_row_of_ones(matrix):
    lil_mat = matrix.toarray()
    (r,c) = lil_mat.shape
    lil_mat[r-1, :] = 1
    return csr_matrix(lil_mat, dtype=np.float64)

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
    # apply sigmoid function
    sig = 1/(1+np.exp(-Y))
    idxs = np.argmax(sig, axis=1)
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

    training_old, training = create_scipy_csr('sparse_training_lr')
    test_data, X = create_scipy_csr('sparse_testing_lr')

    # if a column of 0s got truncated due to CRS format. Then add back in?
    (xr,xc) = X.get_shape()
    if (xc == 61188):
        X.resize((xr,xc+1))

    # Initialize weight and delta matrix
    W  = csr_matrix((num_classes, 61188+1), dtype=np.float64)
    delta = build_delta_matrix(training_old)

    # Remove column with class values from training data
    mat_size = training.get_shape()
    training.resize((mat_size[0], mat_size[1]-1))

    # normalize by columns
    training_norm = normalize(training, norm='max',axis=0)

    # Call logistic regression
    W = logistic_regression(W, training_norm, delta, eta, lam)

    print("W=", W.toarray())

    X = normalize(X, norm='max',axis=0)
    Y = W * X.transpose()
    classify(Y.transpose().toarray())
    score = util.get_accuracy_score('test_col.csv', 'lr_output.csv')
    print(score)
 
