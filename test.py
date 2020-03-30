import logistic_regression as lg
from scipy.sparse import csr_matrix
import numpy as np
from util import Sparse_CSR
import util
from numpy import arange
import matplotlib.pyplot as plt

import main



def plot_naive_bayes(step_size):
    beta = []
    acc  = []
    for i in arange(0.00001, 1, step_size):
        beta.append(i)
        acc.append(main.multi_classification_nb(i))
    fig, ax = plt.subplots()
    ax.plot(beta, acc)
    ax.set(xlabel='Beta', ylabel='Accuracy',
           title='Naive Bayes: Beta 0.00001 to 1 with ' + str(step_size) +' step')
    ax.grid()
    fig.savefig("naive-bayes.png")
    plt.show()


def perform_other_tests():
    mat = [[1,3,0,3],[2,7,0,1],[0,0,1,2]]
    print(mat)
    csr_mat = csr_matrix(mat, dtype=np.float64)
    #in_house_csr = Sparse_CSR([1,3,3,2,7,1,1,2],  [0,3,6], [0,1,3,0,1,3,2,3])

    #ones = lg.add_row_of_ones(csr_mat)
    ##print(ones.toarray())

    #norm = lg.normalize_matrix(csr_mat)
    ##print(norm.toarray())

    #delta = lg.build_delta_matrix(in_house_csr)
    #print(delta.toarray())

    x = lg.row_normalize_matrix(csr_mat)
    print(x.toarray())

    mat = util.process_csv_ones('test.csv')
    print("DATA", mat.data)
    print("cols", mat.cols)
    print("rows", mat.rows)


if (__name__ == '__main__'):
    #perform_other_tests()
    plot_naive_bayes(0.00001)

    
