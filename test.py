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
    ax.semilogx(beta, acc)
    #ax.plot(beta, acc)
    ax.set(xlabel='Beta', ylabel='Accuracy',
           title='Naive Bayes: Beta 0.00001 to 1 with ' + str(step_size) +' step')
    ax.grid()
    fig.savefig("naive-bayes.png")
    plt.show()

def plot_logistic_regression(step_eta, step_lam):
    eta = []
    lam = []
    acc = []
    print('test')
    for i in arange(0.001, 0.01, step_eta):
        for j in arange(0.001, 0.01, step_lam):
            eta.append(i)
            lam.append(j)
            print('eta = ', i, 'lam = ', j)
            acc.append(lg.multi_classification_lr(i, j))
    fig, ax = plt.subplots()
    #ax.semilogx(eta, lam, acc)
    ax.plot(eta, lam, acc)
    ax.set(xlabel='Eta and Lambda', ylabel='Accuracy',
           title='Logistic Regression: Eta 0.001 to 0.01 with ' + str(step_eta) +' step, Lambda 0.001 to 0.01 with ' + str(step_lam) + ' step')
    ax.grid()
    fig.savefig("logistic-regression.png")
    plt.show()



def perform_other_tests():

    mat = [[1,3,0,0],[2,7,0,0],[0,0,1,0]]
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
    #plot_naive_bayes(0.001)
    plot_logistic_regression(0.01, 0.01)
    #mat = util.process_csv('test.csv',0)


    #mat = [[1,3,0,0],[2,7,0,0],[0,0,1,0]]
    #mat_csr = csr_matrix(mat)
    #print(mat_csr.get_shape())
    #print(mat_csr.toarray())

    #mat_csr.resize(3,6)
    #print(mat_csr.get_shape())
    #print(mat_csr.toarray())
