from util import Sparse_CSR
from scipy.sparse import csr_matrix
import pickle
import sys
import numpy as np


def create_scipy_csr():
    file1 = open('sparse_training_ones', 'rb')
    matrix = pickle.load(file1)
    file1.close()
    matrix_scipy = csr_matrix((matrix.data, matrix.cols, matrix.rows))
    return matrix, matrix_scipy

def probability_values(W, X):
    matrix = W * X.transpose()
    return matrix.expm1() # This is exponential - 1, may make a difference. ****

def build_delta_matrix(matrix):
    data = []
    row  = []
    col  = []
    column = 0
    for i in range(1, 12001):
        classification = matrix.last_col_value(i)
        data.append(1)
        row.append(classification - 1)
        col.append(column)
        column += 1
    delta = csr_matrix((data, (row, col)))
    return delta


def logistic_regression(W, X, Del, eta, lam):
    W1 = W
    #print('length of W', W.get_shape())
    #print('length of X', X.get_shape())
    #print('length of Delta', Del.get_shape())
    for i in range(0, 1000):
        WX = probability_values(W1, X)
        W1 = W1 + eta * ((Del - WX) * X - (lam * W1))
    return W1



if (__name__ == '__main__'):

    if len(sys.argv) < 3:
        print('Must enter commandline arguments <Eta> <Lambda>')
        print("Eta:    0.01 to 0.001")
        print("Lambda: 0.01 to 0.001")
        exit(0)

    eta = float(sys.argv[1])
    lam = float(sys.argv[2])

    mat, matrix = create_scipy_csr()
    #print(mat.data)
    obj = np.zeros((20, 61189 + 1))
    obj2 = obj.tolist()
    W = csr_matrix(obj2)
    delta = build_delta_matrix(mat)
    #print(delta.toarray())
    out = logistic_regression(W, matrix, delta, eta, lam) # Pick random eta and lam.
    #print(matrix.getrow(0))
    print(out)
    #print(mat.get_row(0))
