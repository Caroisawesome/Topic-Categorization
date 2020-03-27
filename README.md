# Topic-Categorization

This repo contains the code to perform a topic categorization task, using both Multinomial Naive Bayes, and Logistic Regression.

## To run the program

Clone repo: `git clone `

### Pre-process the data
Run `python3 util.py` first in order to process and parse the data from `data/training.csv` and `data/testing.csv`. This will produce serialized CRS sparse matrices to store the training and testing data, as well as a csv file `test_col.csv` which contains the correct answers associated with the testing data.

### Naive Bayes

Run `python3 main.py {beta}` where `beta` is a term between  0.1 and 0.00001. This will output an accuracty score between 0 and 1, and will also output a file `output_nb.csv` containing the classified values.

### Logistic Regression

Run `python3 logistic_regression {eta} {lambda}` where `eta` is a value between 0.01 and 0.001, and where `lambda` is between 0.1 and 0.001. This will return an accuracy score between 0 and 1, and will output a file `output_lr.csv` containing the classified values. 

