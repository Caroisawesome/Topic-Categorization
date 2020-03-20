from scipy.sparse import csr_matrix
import numpy as np
#import pandas
import csv
import pickle

class Sparse_CSR:
    def __init__(self, data, rows, cols):
        self.data = data
        self.rows = rows
        self.cols = cols
        self.num_rows = len(rows)
        self.len_data = len(data)

    def access_element(self, row, col):
        row_val  = self.rows[row]
        data_idx = row_val
        cols     = self.cols[row_val]
        while cols <= col:
            if cols == col:
                return self.data[data_idx]
            data_idx += 1
            cols = self.cols[data_idx]
        return 0

    def last_col_value(self, row):
        idx = self.get_idx_last_item_in_row(row)
        return self.data[idx]

    def get_idx_last_item_in_row(self, row):
        if self.num_rows == row + 1:
            return self.len_data - 1
        else:
            return self.rows[row + 1] - 1

    def get_row(self, row):
        out = []
        for i in range(self.rows[row], self.get_idx_last_item_in_row(row)+1):
            out.append(self.data[i])
        return out

def partition_csv(name, first):
    """

    Take the name for a CSV along with the size of the first partition 
    and break the csv into two files.

    """
    with open(name, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # Read into list of lists
        tmp = list(list(rec) for rec in csv.reader(csvfile, delimiter=','))
        split_a = tmp[:first]
        split_b = tmp[first:]
        print(len(split_a))
        print(len(split_b))


def write_csv(name, data):
    """

    Write data to csv file. Takes desired filename and data to be written.

    """
    file = name + '.csv'
    with open(file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(['id', 'class'])
        for x in data:
            writer.writerow(x)

def process_csv(filename, ones):
    """

    Function for reading csv data files.
    Takes filename as argument. Uses CSR format

    """
    tmp  = []
    data = []
    cols = []
    rows = []
    idx_data = 0
    flag = False
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # reads csv into a list of lists
        tmp = list(list(rec) for rec in csv.reader(csvfile, delimiter=','))
        if ones == True:
            tmp[:0] = [np.ones(len(tmp[0]))]
        #yield next(reader)
        for row in tmp:
            for i in range(1, len(row)):
                val = int(row[i])
                if val > 0:
                    data.append(val)
                    cols.append(i)
                    if not flag:
                        idx_data = len(data) - 1
                        flag = True
                        rows.append(idx_data)
            flag = False
        #    data.append((row[0], row[1], row[2]))
    #for d in data: #    print(d)
    matrix = Sparse_CSR(data, rows, cols)
    return matrix

#if (__name__ == '__main__'):
#    #print("main")
#
#    # Import data for Logistic Regression
#    matrix_lr = process_csv('data/training.csv', True)
#    matrix_lr_test = process_csv('data/testing.csv', True)
#
#    file = open('sparse_training_lr', 'wb')
#    file2 = open('sparse_testing_lr', 'wb')
#
#    pickle.dump(matrix_lr, file)
#    pickle.dump(matrix_lr_test, file2)
#
#    file.close()
#    file2.close()
#
#    # Import data for Naive Bayes
#    matrix_nb = process_csv('data/training.csv', False)
#    matrix_nb_test = process_csv('data/testing.csv', False)
#
#    file_nb = open('sparse_training_nb', 'wb')
#    file2_nb = open('sparse_testing_nb', 'wb')
#
#    pickle.dump(matrix_nb, file)
#    pickle.dump(matrix_nb_test, file2)
#
#    file_nb.close()
#    file2_nb.close()

