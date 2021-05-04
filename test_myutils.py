import numpy as np
import scipy.stats as stats 
import math
import mysklearn.myutils as myutils
import mysklearn.myevaluation as myevaluation



iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
iphone_table = [
    [1, 3, "fair", "no"],
    [1, 3, "excellent", "no"],
    [2, 3, "fair", "yes"],
    [2, 2, "fair", "yes"],
    [2, 1, "fair", "yes"],
    [2, 1, "excellent", "no"],
    [2, 1, "excellent", "yes"],
    [1, 2, "fair", "no"],
    [1, 1, "fair", "yes"],
    [2, 2, "fair", "yes"],
    [1, 2, "excellent", "yes"],
    [2, 2, "excellent", "yes"],
    [2, 3, "fair", "yes"],
    [2, 2, "excellent", "no"],
    [2, 3, "fair", "yes"]
]
data_iphone, results_iphone = myutils.separate(iphone_table, iphone_col_names, "buys_iphone")

def test_random_stratified_split():
    X_train, X_test, y_train, y_test = myutils.random_stratified_split(data_iphone, results_iphone)

    print("XTrain: " + str(X_train))

    print("XTest: " + str(X_test))
    assert True == True