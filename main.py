import pickle
import util


if (__name__ == '__main__'):
    file = open('sparse_testing', 'rb')
    matrix = pickle.load(file)
    file.close()
    val = matrix.access_element(6, 10)
    print('Value is: ' + str(val))

