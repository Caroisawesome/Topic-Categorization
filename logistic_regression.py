from util import Sparse_CSR
from scipy.sparse import csr_matrix
import pickle
import numpy as np


def create_scipy_csr():
    file1 = open('sparse_training', 'rb')
    matrix = pickle.load(file1)
    file1.close()
    matrix_scipy = csr_matrix((matrix.data, matrix.cols, matrix.rows))
    return matrix, matrix_scipy

def probability_values(W, X):
    matrix = W.multiply(X.transpose(axes=None,copy=True))
    return matrix.expm1() # This is exponential - 1, may make a difference. ****

def build_delta_matrix(matrix):
    data = []
    row  = []
    col  = []
    column = 0
    for i in range(0, 12000):
        classification = matrix.last_col_value(i)
        data.append(1)
        row.append(classification - 1)
        col.append(column)
        column += 1
    delta = csr_matrix((data, (row, col)))
    return delta


#def logistic_regression(W, X, eta, lam):
#    for i in range(0, 1000):



if (__name__ == '__main__'):
    mat, matrix = create_scipy_csr()
    W = np.zeros((20,61189))
    delta = build_delta_matrix(mat)
    print(delta.toarray())
    #logistic_regression(W, matrix.transpose(axes=None, copy=True), 0.01, 0.01) # Pick random eta and lam.
    #print(matrix.getrow(0))
    #print(mat.get_row(0))
