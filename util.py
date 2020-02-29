from scipy import sparse
import numpy as np
#import pandas
import csv

#def generate_coo_sparse_matrix():

def process_csv(filename):
    """

    Function for reading training data csv files.
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
        for row in tmp:
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
        #    data.append((row[0], row[1], row[2]))
    #for d in data: #    print(d)
    return [data, cols, rows]

