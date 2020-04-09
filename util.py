from scipy.sparse import csr_matrix
import numpy as np
import csv
import pickle
import random

class Sparse_CSR:
    """

    Class for representing a matrix in Sparse CSR format

    """
    def __init__(self, data, rows, cols):
        self.data = data
        self.rows = rows
        self.cols = cols
        self.num_rows = len(rows)
        self.len_data = len(data)

    def access_element(self, row, col):
        """

        Returns the data element at index (row, column) in the matrix.

        """
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
        """

        Returns the data value of the last item in the row.
        row: index of the row

        """
        idx = self.get_idx_last_item_in_row(row)
        return self.data[idx]

    def get_idx_last_item_in_row(self, row):
        """

        Returns the index of where the last item in the row is stored in the data array.
        row: index of the row

        """
        if self.num_rows == row + 1:
            return self.len_data - 1
        else:
            return self.rows[row + 1] - 1

    def get_row(self, row):
        """

        Returns an array containing the data values corresponding with a particular row,
             as they are stored in sparse CRS format. 

        """
        out = []
        for i in range(self.rows[row], self.get_idx_last_item_in_row(row)+1):
            out.append(self.data[i])
        return out


def partition_csv_alt(name, first):
    print("partitioning")
    answers = []
    with open('training_new.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')

        with open('testing_new.csv','w') as testcsv:
            writer_test = csv.writer(testcsv, delimiter=',')

            with open(name, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for i, row in enumerate(reader):
                    # overwrite id values with a column of 1s
                    row[0] = 1
                    #if random.random() < 0.85:
                    if i < first:
                        writer.writerow(row)
                    else:
                        class_val = row.pop()
                        writer_test.writerow(row)
                        answers.append([int(class_val)])

    write_csv_new('test_col', answers)


def partition_csv(name, first):
    """

    Take the name for a CSV along with the size of the first partition
    and break the csv into two files.

    """
    print("opened csv file")
    # Read into list of lists
    with open(name, 'r') as csvfile:
        reader = csv.reader(csvfile)
        tmp = list(list(rec) for rec in csv.reader(csvfile, delimiter=','))
        #tmp = shuffle(tmp)
        print("read in data", len(tmp))
        split_a = tmp[:first]
        split_b = tmp[first:]
        split_c = [[]]
        last = len(split_b[0]) - 1
        for i in range(0, len(split_b)):
            split_c.append([split_b[i][last]])
            del split_b[i][last]
        write_csv_new('training_new', split_a)
        write_csv_new('testing_new',  split_b)
        write_csv_new('test_col',     split_c)
        #print(len(split_a[0]))
        #print(len(split_b[0]))
        #print(split_c)


def write_csv(name, data):
    """

    Write data to csv file. Takes desired filename and data to be written.

    """
    print("in write csv function")
    file = name + '.csv'
    with open(file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(['id', 'class'])
        for x in data:
            #print(x)
            writer.writerow(x)

def write_csv_new(name, data):
    """

    Write data to csv file. Takes desired filename and data to be written. No header

    """
    file = name + '.csv'
    with open(file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        for x in data:
            writer.writerow(x)


def process_csv(filename, start_col):
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
        #yield next(reader)
        print("columns", len(tmp[0]))
        for row in tmp:
            for i in range(start_col, len(row)):
                if i == 0:
                    val = 1
                else:
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
    matrix_scipy = csr_matrix((matrix.data, matrix.cols, matrix.rows), dtype=np.float64)
    print("matrix size", matrix_scipy.get_shape())
    return matrix


def get_accuracy_score(correct_data, classified_data):
    """

    Returns the accuracy score between 0 and 1, between two answer sets.
    correct_data:  The filename containing of the file the correctly classified values
    classified_data: The filename of the file containing the "guess" values

    """
    actual = []
    guess = []
    with open(correct_data, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # Read into list of lists
        tmp = list(list(rec) for rec in csv.reader(csvfile, delimiter=','))
        tmp.pop(0)
        for row in tmp:
            actual.append(row[0])

    with open(classified_data,'r') as csvfile:
        reader = csv.reader(csvfile)
        # Read into list of lists
        tmp = list(list(rec) for rec in csv.reader(csvfile, delimiter=','))
        for row in tmp:
            guess.append(row[1])

    num_correct = 0
    for i in range(1,len(actual)):
        if (actual[i-1] == guess[i]):
            num_correct+=1
    return num_correct/len(actual)


def process_data_for_lr():
    # Import data for Logistic Regression
    matrix_lr = process_csv('training_new.csv',0)
    matrix_lr_test = process_csv('testing_new.csv',0)

    file = open('sparse_training_lr', 'wb')
    file2 = open('sparse_testing_lr', 'wb')

    pickle.dump(matrix_lr, file)
    pickle.dump(matrix_lr_test, file2)

    file.close()
    file2.close()

def process_data_for_nb():
    # Import data for Naive Bayes
    matrix_nb = process_csv('training_new.csv',1)
    matrix_nb_test = process_csv('testing_new.csv',1)

    file_nb = open('sparse_training_nb', 'wb')
    file2_nb = open('sparse_testing_nb', 'wb')

    pickle.dump(matrix_nb, file_nb)
    pickle.dump(matrix_nb_test, file2_nb)

    file_nb.close()
    file2_nb.close()


if (__name__ == '__main__'):
    partition_csv('data/training.csv', 10000)
    process_data_for_lr()
    process_data_for_nb()
