import mysklearn.myutils as myutils
import operator
import random
import mysklearn.mypytable as mypytable
import os


class MySimpleLinearRegressor:
    """Represents a simple linear regressor.
    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b
    Notes:
        Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    
    def __init__(self, slope=None, intercept=None, printer = False):
        """Initializer for MySimpleLinearRegressor.
        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
            printer(bool): print out linear regression statistics (defaults to False)
        """
        self.slope = slope 
        self.intercept = intercept
        self.printer = printer

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], 
                                           [1], 
                                           [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train) 
                The shape of y_train is n_train_samples
        """
        self.slope,self.intercept,_,_ = myutils.calculateLeastSquares(myutils.nestedListToList(X_train), y_train, self.printer)
        pass

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        return [self.slope * ii + self.intercept for ii in myutils.nestedListToList(X_test)]



class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.
    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.
        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train 
        self.y_train = y_train 

    def kneighbors(self, X_test):
        """Determines the k closest neighbors of each test instance.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances 
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        indices = []
        distances = []
        for jj in X_test:
            ind= []
            distance = []
            for i, instance in enumerate(self.X_train):
                # append the class label
                instance.append(self.y_train[i])
                # append the original row index
                instance.append(i)
                dist = myutils.compute_euclidean_distance(instance[:-2], jj)
                instance.append(dist)
            
            # sort train by distance
            train_sorted = sorted(self.X_train, key=operator.itemgetter(-1))
            top_k = train_sorted[:self.n_neighbors]
            for instance in top_k:
                ind.append(instance[-2])
                distance.append(instance[-1])

            # should remove the extra values at the end of each instance
            for instance in self.X_train:
                del instance[-3:]
            distances.append(distance)
            indices.append(ind) 
        return distances, indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        result = []
        _, indices = self.kneighbors(X_test)
        for lis in indices:
            temp = []
            for jj in lis:
                temp.append(self.y_train[jj])
            result.append(myutils.findMostFrequent(temp))
        return result

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.
    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        priors(list): The prior probabilities computed for each
            label in the training set. parallel to labels list
        posteriors(nested dictionary): The posterior probabilities computed for each
            attribute value/label pair in the training set.
    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.X_train = None 
        self.y_train = None
        self.priors = []
        self.posteriors = {}

        self.classes = []

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.
        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.X_train = X_train
        self.y_train = y_train

        classNames, classTables = myutils.group_by(X_train, y_train)

        [self.classes.append(item) for item in y_train if item not in self.classes]
        for x in self.classes:
            self.priors.append(y_train.count(x)/len(y_train))
            self.posteriors[x] = {}

        for ii in range(len(X_train)):
            for jj in range(len(X_train[0])):
                label = "att" + str(jj) + "=" + str(X_train[ii][jj])
                for aClass in self.classes:
                    self.posteriors[aClass][label] = 0

        for ii, aClass in enumerate(classNames):
            for jj in range(len(classTables[ii][0])):
                for kk in range(len(classTables[ii])):
                    label = "att" + str(jj) + "=" + str(classTables[ii][kk][jj])
                    if label in self.posteriors[aClass].keys():
                        self.posteriors[aClass][label] += 1
                    else:
                        print("Not Allowed")
        for key, value in self.posteriors.items():
            for akey in value:
                value[akey] /= len(classTables[classNames.index(key)])
        pass

    def predict(self, X_test):
        """Makes predictions for test instances in X_test
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        answers = []
        for instance in X_test:
            mathematics = [x for x in self.priors]
            for jj, feature in enumerate(instance):
                label = "att" + str(jj) + "=" + str(feature)
                if label in self.posteriors[self.classes[0]].keys():
                    for kk in range(len(self.classes)):
                        mathematics[kk] *= self.posteriors[self.classes[kk]][label]
                else:
                    print("No")
            answers.append(self.classes[mathematics.index(max(mathematics))])
        
        return answers

class MyZeroRClassifier:
    """Represents a "zero rule" classifier which always predicts the most common class label in the training set.
    Attributes:
        result(obj): most common class in y_train
    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        self.result = None
    
    def fit(self, y_train):
        """Finds most common element in y_train and saves that as result
        Args:
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.result = myutils.findMostFrequent(y_train)

    def predict(self, X_test):
        """returns result calculated in fit
        Args:
            X_test(list of list): The testing x values
                The shape of X_test is (n_test_samples, n_features)
        """
        return [self.result for instance in X_test]

class MyRandomClassifier:
    """Represents a random class choice based on a weighted average of all the elements in y_train.
    Attributes:
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        options(list of obj): takes all the classes in y_train and removes copies
        frequencies(list of int): parallel list to options where each corresponding index holds the frequency
    """
    def __init__(self):
        self.y_train = None
        self.options = []
        self.frequencies = []
    
    def fit(self, y_train):
        """calculates frequencies for y_train
        Args:
            y_train(list of obj): The training y values
                The shape of y_train is n_test_samples
        """
        self.y_train = y_train
        [self.options.append(item) for item in y_train if item not in self.options]
        self.frequencies = [y_train.count(item)/len(y_train) for item in self.options]
    def predict(self):
        """returns result calculated in fit
        """
        return random.choices(self.options, weights=self.frequencies, k=1)

        
class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.
    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None 
        self.y_train = None
        self.tree = None
        self.domain = {}
        self.header = []

    def partition_instances(self, instances, split_attribute):
    # this is a group by split_attribute's domain, not by
    # the values of this attribute in instances
    # example: if split_attribute is "level"
        attribute_domain = sorted(self.domain[split_attribute]) # ["Senior", "Mid", "Junior"]
        attribute_index = self.header.index(split_attribute) # 0
        # lets build a dictionary
        partitions = {} # key (attribute value): value (list of instances with this attribute value)
        # task: try this!
        for attribute_value in attribute_domain:
            partitions[attribute_value] = []
            for instance in instances:
                if instance[attribute_index] == attribute_value:
                    partitions[attribute_value].append(instance)
        return partitions

    def tdidt(self, current_instances, available_attributes):
        # basic approach (uses recursion!!):

        # select an attribute to split on
        split_attribute = myutils.select_attribute(current_instances, available_attributes, self.header)
        available_attributes.remove(split_attribute)
        
        # cannot split on the same attribute twice in a branch
        # recall: python is pass by object reference!!
        tree = ["Attribute", split_attribute]

        prevPartition = current_instances
        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions = self.partition_instances(current_instances, split_attribute)

        # for each partition, repeat unless one of the following occurs (base case)
        for attribute_value, partition in partitions.items():
            value_subtree = ["Value", attribute_value]
            # TODO: appending leaf nodes and subtrees appropriately to value_subtree
            
            #    CASE 1: all class labels of the partition are the same => make a leaf node
            if len(partition) > 0 and myutils.all_same_class(partition):
                value_subtree.append(["Leaf", partition[0][-1], len(partition), len(current_instances)])

            #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
            elif len(partition) > 0 and len(available_attributes) == 0:
                [majority, count, total] = myutils.compute_partition_stats(partition)
                value_subtree.append(["Leaf", majority, count, total])

            #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
            elif len(partition) == 0:
                [majority, count, total] = myutils.compute_partition_stats(prevPartition)
                value_subtree.append(["Leaf", majority, count, total])
            else: # all base cases are false... recurse!!
                subtree = self.tdidt(partition, available_attributes.copy())
                value_subtree.append(subtree)
            tree.append(value_subtree)
                # need to append subtree to value_subtree and appropriately append value subtree to tree
        return tree

    def tdidt_predict(self,header, tree, instance):
        info_type = tree[0]
        if info_type == "Attribute":
            attribute_index = self.header.index(tree[1])
            instance_value = instance[attribute_index]
            # now I need to find which "edge" to follow recursively
            for i in range(2, len(tree)):
                value_list = tree[i]
                if value_list[1] == instance_value:
                    # we have a match!! recurse!!
                    return self.tdidt_predict(header, value_list[2], instance)
        else: # "Leaf"
            return tree[1] # leaf class label

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.
        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        self.header = ["att" + str(ii) for ii in range(len(X_train[0]))]
        for jj in range(len(X_train[0])):
            temp = []
            for ii in range(len(X_train)):
                if X_train[ii][jj] not in temp:
                    temp.append(X_train[ii][jj])
            self.domain[self.header[jj]] = temp
        
        # my advice is to stitch together X_train and y_train
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        # initial call to tdidt current instances is the whole table (train)
        available_attributes = self.header.copy() # python is pass object reference
        self.tree = self.tdidt(train, available_attributes)
        pass
        


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        results = []
        for instance in X_test:
            results.append(self.tdidt_predict(self.header, self.tree, instance))
        return results

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.
        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        myutils.recurse_tree(self.header, self.tree, class_name, attribute_names, "IF ")    
        pass

class MyRandomForestClassifier:
    """Represents a decision tree classifier.
    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, M, N, F):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.M = M
        self.N = N
        self.F = F
        self.best_M_trees = []
        self.domain = {}
        self.header = []

    def partition_instances(self, instances, split_attribute):
    # this is a group by split_attribute's domain, not by
    # the values of this attribute in instances
    # example: if split_attribute is "level"
        attribute_domain = sorted(self.domain[split_attribute]) # ["Senior", "Mid", "Junior"]
        attribute_index = self.header.index(split_attribute) # 0
        # lets build a dictionary
        partitions = {} # key (attribute value): value (list of instances with this attribute value)
        # task: try this!
        for attribute_value in attribute_domain:
            partitions[attribute_value] = []
            for instance in instances:
                if instance[attribute_index] == attribute_value:
                    partitions[attribute_value].append(instance)
        return partitions

    def tdidt(self, current_instances, available_attributes):
        # basic approach (uses recursion!!):
        # select an attribute to split on
        split_attribute = myutils.select_attribute(current_instances, available_attributes, self.header)
        available_attributes.remove(split_attribute)
        
        # cannot split on the same attribute twice in a branch
        # recall: python is pass by object reference!!
        tree = ["Attribute", split_attribute]

        prevPartition = current_instances
        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions = self.partition_instances(current_instances, split_attribute)

        # for each partition, repeat unless one of the following occurs (base case)
        for attribute_value, partition in partitions.items():
            value_subtree = ["Value", attribute_value]
            # TODO: appending leaf nodes and subtrees appropriately to value_subtree
            
            #    CASE 1: all class labels of the partition are the same => make a leaf node
            if len(partition) > 0 and myutils.all_same_class(partition):
                value_subtree.append(["Leaf", partition[0][-1], len(partition), len(current_instances)])

            #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
            elif len(partition) > 0 and len(available_attributes) == 0:
                [majority, count, total] = myutils.compute_partition_stats(partition)
                value_subtree.append(["Leaf", majority, count, total])

            #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
            elif len(partition) == 0:
                [majority, count, total] = myutils.compute_partition_stats(prevPartition)
                value_subtree.append(["Leaf", majority, count, total])
            else: # all base cases are false... recurse!!
                if len(available_attributes) > self.F:
                    subtree = self.tdidt(partition, random.sample(available_attributes.copy(), self.F))

                else:
                    subtree = self.tdidt(partition, available_attributes.copy())
                value_subtree.append(subtree)
            tree.append(value_subtree)
                # need to append subtree to value_subtree and appropriately append value subtree to tree
        return tree

    def tdidt_predict(self,header, tree, instance):
        info_type = tree[0]
        if info_type == "Attribute":
            attribute_index = self.header.index(tree[1])
            instance_value = instance[attribute_index]
            # now I need to find which "edge" to follow recursively
            for i in range(2, len(tree)):
                value_list = tree[i]
                if value_list[1] == instance_value:
                    # we have a match!! recurse!!
                    return self.tdidt_predict(header, value_list[2], instance)
        else: # "Leaf"
            return tree[1] # leaf class label


    def fit(self, X_train, y_train, X_test, y_test):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.
        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        trees = []
        for jj in range(self.N):
            tree = None
            self.header = ["att" + str(ii) for ii in range(len(X_train[0]))]
            for jj in range(len(X_train[0])):
                temp = []
                for ii in range(len(X_train)):
                    if X_train[ii][jj] not in temp:
                        temp.append(X_train[ii][jj])
                self.domain[self.header[jj]] = temp
            
            # my advice is to stitch together X_train and y_train
            train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
            bootstrapped_train = myutils.compute_bootstrapped_sample(train)
            # initial call to tdidt current instances is the whole table (train)
            available_attributes = self.header.copy()
            tree = self.tdidt(bootstrapped_train, available_attributes)
            trees.append(tree)

        performances = []
        for ii, tree in enumerate(trees):
            counter = 0
            for jj, instance in enumerate(X_test):
                if self.tdidt_predict(self.header, tree, instance) == y_test[jj]:
                    counter+=1
            performances.append(counter/len(y_test))
        
        # Thank you stackoverflow
        # https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list

        sortedtrees = [x for _,x in sorted(zip(performances, trees))]

        self.best_M_trees = sortedtrees[:self.M]
        pass
        


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        
        results = []
        for instance in X_test:
            treeResults = []
            for tree in self.best_M_trees:
                treeResults.append(self.tdidt_predict(self.header, tree, instance))
            results.append(myutils.findMostFrequent(treeResults))
            print(treeResults)
        return results
            

        

        
    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.
        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        for tree in self.best_M_trees:
            myutils.recurse_tree(self.header, tree, class_name, attribute_names, "IF ")  
            print("DONE")  
        pass