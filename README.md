# Topic-Categorization

This repo contains the code to perform a topic categorization task, using both Multinomial Naive Bayes, and Logistic Regression. Given a set of words which contain the number of times each word appears in the document, we identify one of twenty topics that correspond with the set of words. Using the Naive Bayes method we are able to get a top accuracy score of 88\% when classifying new data, and with Logistic Regression we are able to get an accuracy score of 81\%. The results can be seen in the `Naive_Bayes_and_Logistic_Regression.pdf` document.
`

### To run the program

Clone repo: `git clone `

#### Pre-process the data
Run `python3 util.py` first in order to process and parse the data from `data/training.csv` and `data/testing.csv`. This will produce serialized CRS sparse matrices to store the training and testing data, as well as a csv file `test_col.csv` which contains the correct answers associated with the testing data.

#### Naive Bayes

Run `python3 main.py {beta}` where `beta` is a term between  0.1 and 0.00001. This will output an accuracty score between 0 and 1, and will also output a file `output_nb.csv` containing the classified values.

#### Logistic Regression

Run `python3 logistic_regression {eta} {lambda}` where `eta` is a value between 0.01 and 0.001, and where `lambda` is between 0.1 and 0.001. This will return an accuracy score between 0 and 1, and will output a file `output_lr.csv` containing the classified values. 

### To run Tests and Plots

Run `python util.py` and then `python test.py`.

To changes the types of tests being run, modify the main method in `test.py`.

### To plot the confusion matrix 

After running logistic regression, then run `python confusion_matrix.py`.




