import pickle
from util import Sparse_CSR


if (__name__ == '__main__'):
    file = open('sparse_testing', 'rb')
    matrix = pickle.load(file)
    file.close()
    val = matrix.access_element(7, 10)
    print('Value is: ' + str(val))

