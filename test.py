import logistic_regression as lg
from scipy.sparse import csr_matrix
import numpy as np
from util import Sparse_CSR

if (__name__ == '__main__'):

    mat = [[1,1,1,1],[2,7,0,1],[0,0,1,2]]
    print(mat)

    csr_mat = csr_matrix(mat, dtype=np.float64)
    in_house_csr = Sparse_CSR([1,1,1,1,2,7,1,1,2],  [0,4,7], [0,1,2,3,0,1,3,2,3])

    ones = lg.add_row_of_ones(csr_mat)
    #print(ones.toarray())

    norm = lg.normalize_matrix(csr_mat)
    print(norm.toarray())

    delta = lg.build_delta_matrix(in_house_csr)
    #print(delta.toarray())
