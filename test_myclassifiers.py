import numpy as np
import scipy.stats as stats 
import math
import mysklearn.myutils as myutils

from mysklearn.myclassifiers import MySimpleLinearRegressor, MyKNeighborsClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier, MyRandomForestClassifier

interview_header = ["level", "lang", "tweets", "phd", "interviewed_well"]
interview_table = [
        ["Senior", "Java", "no", "no", "False"],
        ["Senior", "Java", "no", "yes", "False"],
        ["Mid", "Python", "no", "no", "True"],
        ["Junior", "Python", "no", "no", "True"],
        ["Junior", "R", "yes", "no", "True"],
        ["Junior", "R", "yes", "yes", "False"],
        ["Mid", "R", "yes", "yes", "True"],
        ["Senior", "Python", "no", "no", "False"],
        ["Senior", "R", "yes", "no", "True"],
        ["Junior", "Python", "yes", "no", "True"],
        ["Senior", "Python", "yes", "yes", "True"],
        ["Mid", "Python", "no", "yes", "True"],
        ["Mid", "Java", "yes", "no", "True"],
        ["Junior", "Python", "no", "yes", "False"]
    ]

M = 7
N = 20
F = 2
interviewTest = MyRandomForestClassifier(M, N, F)
interviewData, interviewClasses = myutils.separate(interview_table, interview_header, "interviewed_well")


def test_decision_tree_classifier_fit():

    X_train, y_train, X_test, y_test = myutils.random_stratified_split(interviewData, interviewClasses)
    interviewTest.fit(X_train, X_test, y_train, y_test)
    assert len(interviewTest.best_M_trees) == M

def test_decision_tree_classifier_predict():
    results = interviewTest.predict([["Junior", "Java", "yes", "no"], ["Junior", "Java", "yes", "yes"]])
    actual = ['True', 'False']
    assert len(results) == len(actual)
    print(str(results) + " vs " + str(actual))