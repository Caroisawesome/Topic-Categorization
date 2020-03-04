import pickle
import numpy as np
from util import Sparse_CSR


def create_probability_matrix_2(crs_matrix):
    M = np.zeros((20,61188))
    class_totals = np.zeros(20)

    for row in range(0, crs_matrix.num_rows):
        row_start = crs_matrix.rows[row]
        row_end = crs_matrix.get_idx_last_item_in_row(row)
        class_val = crs_matrix.data[row_end] - 1
        class_totals[class_val]+=1
        for i in range(row_start, row_end):
            col = crs_matrix.cols[i]
            M[class_val][col] += crs_matrix.data[i]

    return convert_matrix_to_CRS(M)

def convert_matrix_to_CRS(matrix):
    data = []
    cols = []
    rows = []
    idx_data = 0
    flag = False

    for row in matrix:
        for i in range(0, len(row)):
            val = int(row[i])
            if val > 0:
                data.append(val)
                cols.append(i)
                if not flag:
                    idx_data = len(data) - 1
                    flag = True
                    rows.append(idx_data)
        flag = False
    matrix = Sparse_CSR(data, rows, cols)
    return matrix


def create_probability_matrix(crs_matrix):

    unique_cols = sorted(list(set(crs_matrix.cols)))
    num_cols = len(unique_cols)
    num_data = crs_matrix.len_data
    col_idx_mapping = {}
    summed_data_matrix =[np.zeros(num_cols) for i in range(20)]
    summed_col_matrix =[np.zeros(num_cols) for i in range(20)]


    print('num unique cols', num_cols)


    # maps column values to a particular index
    for i in range(0,num_cols):
        col_idx_mapping[unique_cols[i]] = i


    for r in range(0, crs_matrix.num_rows):
        start_of_row = crs_matrix.rows[r]
        end_of_row = crs_matrix.get_idx_last_item_in_row(r)

        for c in range(start_of_row, end_of_row):
            row_idx = crs_matrix.data[end_of_row]
            col_idx = col_idx_mapping[crs_matrix.cols[c]]
            data_val = crs_matrix.data[c]
            summed_data_matrix[row_idx][col_idx] += data_val

    for i in range(len(summed_data_matrix)):
        for x in summed_data_matrix[i]:
            if x > 0:
                print(x)
#    print(summed_data_matrix[0])



if (__name__ == '__main__'):
    file = open('sparse_testing', 'rb')
    matrix = pickle.load(file)
    file.close()

    m = create_probability_matrix_2(matrix)
    #print("data", m.data)
    #print("cols", m.cols)
    print("rows", m.rows)

