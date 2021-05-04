import mysklearn.myutils as myutils
import math
import random

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets (sublists) based on a test set size. HOLDOUT METHOD
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting
    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    
    Note:
        Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """

    if random_state is not None:
        random.seed(random_state)
    if shuffle: 
        for jj in range(len(X)):
            # generate a random index to swap the element at i with
            rand_index = random.randrange(0, len(X))
            X[jj], X[rand_index] = X[rand_index], X[jj]
            y[jj], y[rand_index] = y[rand_index], y[jj]

    num_instances = len(X)
    if isinstance(test_size, float):
        test_size = math.ceil(num_instances * test_size)
    split_index = num_instances - test_size 

    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]


def kfold_cross_validation(X, n_splits=5):
    """Split dataset into cross validation folds.
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold
    Notes: 
        The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """

    X_train_folds = []
    X_test_folds = [[] for _ in range(n_splits)]
    for ii in range(len(X)):
        X_test_folds[ii%n_splits].append(ii)

    for jj in range(n_splits):
        X_train_folds.append([item for item in range(len(X)) if item not in X_test_folds[jj]])
    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5):
    """Split dataset into stratified cross validation folds.
    Args:
        X(list of list of obj): The list of instances (samples). 
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X). 
            The shape of y is n_samples
        n_splits(int): Number of folds.
 
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.
    Notes: 
        Loosely based on sklearn's StratifiedKFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """

    X_train_folds = []
    X_test_folds = [[] for _ in range(n_splits)]

    _, group_subtables = myutils.group_by(X,y)

    iterator = 0
    for ii in range(len(group_subtables)):
        for jj, item in enumerate(group_subtables[ii]):
            X_test_folds[iterator%n_splits].append(X.index(item))
            iterator+=1

    for jj in range(n_splits):
        X_train_folds.append([item for item in range(len(X)) if item not in X_test_folds[jj]])

    return X_train_folds, X_test_folds

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix
    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry 
            indicates the number of samples with true label being i-th class 
            and predicted label being j-th class
    Notes:
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = []
    
    for _ in labels:
        temp = []
        for _ in labels:
            temp.append(0)
        matrix.append(temp)
        
    for jj in range(len(y_pred)):
        indexRow = labels.index(y_true[jj])
        indexCol = labels.index(y_pred[jj])
        matrix[indexRow][indexCol] += 1

    return matrix