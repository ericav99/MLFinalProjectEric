import math
import random
import numpy as np
import copy
from statistics import mode
from mysklearn.mypytable import MyPyTable

def separate(data, header, classLabel):
    """ separates the X set from the y set based on classLabel
        Args:
            data(list of list): all the data
            header(list of str): labels for each column
            classLabel(str): label for the chosen class
        Returns:
            data(list of list): all of the X data
            classList(list of obj): all of the y values
    """
    classList = []
    index = header.index(classLabel)
    for row in data:
        classList.append(row[index])
        del row[index]
    return data, classList

def calculateLeastSquares(xCol, yCol, printer):
    """calculates the leastsquares line approximation
        Args:
            xCol (column of floats): values to compute as x values
            yCol (column of floats): values to compute as y values
        Returns:
            m (float): slope of approximate line
            b (float): y-int of approximate line
            r (float): the correlation coefficient
            cov (float): the covariance
    """
    x_mean = sum(xCol)/len(xCol)
    y_mean = sum(yCol)/len(yCol)
    
    num = sum([(xCol[i] - x_mean)*(yCol[i] - y_mean) for i in range(len(xCol))])
    
    m = num / sum([(xCol[i] - x_mean)**2 for i in range(len(xCol))])
    b = y_mean - m * x_mean 
    
    
    r = num / math.sqrt(sum([((xCol[i] - x_mean)**2)for i in range(len(xCol))])*sum([((yCol[i] - y_mean)**2) for i in range(len(xCol))]))
            
    cov = num / len(xCol)
    if printer:    
        print("y = " + str(m) + "x + " + str(b))
        print("Correlation Coefficient: " + str(r))
        print("Covariance: " + str(cov))
    
    return m,b,r,cov

def nestedListToList(nested):
    """flattens a list
        Args:
            nested (list of a list of obj): could be anything
        Returns:
            flattened list
    """
    return [ii for lis in nested for ii in lis]

def compute_bootstrapped_sample(table):
    n = len(table)
    sample = []
    for _ in range(n):
        rand_index = random.randrange(0, n)
        sample.append(table[rand_index])
    return sample

def random_stratified_split(X, y, test_size = 1/3):
    X_test = []
    X_train = []

    y_test = []
    y_train = []
    group_names, group_subtables= group_by(X, y)
    for ii, table in enumerate(group_subtables):
        split_index = math.ceil(len(table) * test_size)
        X_test.append(table[:split_index])
        for jj in range(len(table[:split_index])):
            y_test.append(group_names[ii])
        X_train.append(table[split_index:])
        for jj in range(len(table[split_index:])):
            y_train.append(group_names[ii])

    
    return nestedListToList(X_train), nestedListToList(X_test), y_train, y_test

def group_by(xVals, yVals):
    """collects all xVals by their yVals
        Args:
            xVals (list of obj): attributes
            yVals (list of obj): class names for xVals [parallel to xVals]
        Returns:
            group_names (list of obj): titles of class
            group_subtables (list of list of obj): 
    """
    group_names = list(set(yVals))
    group_subtables = [[] for _ in group_names]
    
    # algorithm: walk through each row and assign it to the appropriate
    # subtable based on its group_by_col_name value
    for ii, identifier in enumerate(yVals):
        group_by_value = identifier
        # which subtable to put this row in?
        group_index = group_names.index(group_by_value)
        group_subtables[group_index].append(copy.deepcopy(xVals[ii]))
    return group_names, group_subtables

def compute_euclidean_distance(v1, v2):
    """checks for equal length and the computer euclidean distance
        Args:
            v1 (float): another number
            v2 (float): a number
        Returns:
            dist (float): distance between v1 and v2
    """
    assert len(v1) == len(v2)    
    if isinstance(v1[0], str) or isinstance(v2[0],str):
        if v1 == v2:
            return 0
        else:
            return 1
    else:
        return np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))

def findMostFrequent(v1):
    """returns the mode of input
    """
    return mode(v1)

def normalize(arr):
    """normalizes an entire array
        Args:
            arr (list of float): values to normalize
        Returns:
            normalized list
    """
    minimum = min(arr)
    arr = [item - minimum for item in arr]
    maximum = max(arr)
    return [item/maximum for item in arr]

def get_DOE_ranking(mpg):
    #Just some bins...
    if mpg <= 13.0:
        return 1
    elif mpg > 13.0 and mpg < 15.0:
        return 2
    elif mpg >= 15.0 and mpg < 17.0:
        return 3
    elif mpg >= 17.0 and mpg < 20.0:
        return 4
    elif mpg >= 20.0 and mpg < 24.0:
        return 5
    elif mpg >= 24.0 and mpg < 27.0:
        return 6
    elif mpg >=27.0 and mpg < 31.0:
        return 7
    elif mpg >=31.0 and mpg < 37.0:
        return 8
    elif mpg >=37.0 and mpg < 45.0:
        return 9
    else:
        return 10

def getNHTSAsizes(weight):
    # more bins
    if weight <= 1999:
        return 1
    elif weight >= 2000 and weight < 2500:
        return 2
    elif weight >= 2500 and weight < 3000:
        return 3
    elif weight >= 3000 and weight < 3500:
        return 4
    elif weight >= 3500:
        return 5

def recurse_tree(header, tree, class_name, attributes, holder):
    info_type = tree[0]
    if info_type == "Attribute":
        if holder != "IF ":
            holder += " AND "
        if attributes is None:
            holder += tree[1]
        else:
            index = header.index(tree[1])
            holder += attributes[index]

        for i in range(2, len(tree)):
            value_list = tree[i]
            temp = holder + " == " + str(value_list[1]) + " "
            recurse_tree(header, value_list[2], class_name, attributes, temp)
    else:
        print(holder + "THEN " + class_name + " == " + str(tree[1]) + "\n")

def select_attribute(instances, available_attributes, original):
    entropyNews =  []
    for index in available_attributes:
        entropyNews.append(compute_entropy(instances, available_attributes, original.index(index)))
    return available_attributes[entropyNews.index(min(entropyNews))]

def compute_entropy(instances, available_attributes, index):
    mypy = MyPyTable(available_attributes, instances)
    classes = mypy.get_column(-1)
    attributes = mypy.get_column(index)
    temp = set(attributes)
    __, tables = group_by(attributes, classes)
    totals = []
    sub_entropies = []
    # get the class counts here
    for jj, element in enumerate(temp):
        totals.append(attributes.count(element))
        # parallel array of counts of each att for each class
        arr = []
        for table in tables:
            arr.append(table.count(element))
        su = 0
        for kk in arr:
            if kk <= 0:
                pass
            else:
                su -= kk/totals[jj]*math.log2(kk/totals[jj])
        su *= totals[jj]/len(attributes)
        sub_entropies.append(su)
    return sum(sub_entropies)


def all_same_class(instances):
    # assumption: instances is not empty and class label is at index -1
    first_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_label:
            return False 
    return True # if we get here, all instance labels matched the first label

def compute_partition_stats(partition):
    arr = [instance[-1] for instance in partition]
    majority = mode(arr)
    count = arr.count(majority)
    total = len(partition)
    return [majority, count, total]

########## Eric myutils
#########

def get_frequencies(MyPyTable, col_name):
    """Gets the frequency and count of a column by name

    Args:
        MyPyTable(MyPyTable): self of MyPyTable
        col_name(str): name of the column

    Returns:
        values, counts (string, int): name of value and its frequency"""

    col = MyPyTable.get_column(col_name)
    values = []
    counts = []

    for value in col:
        if value not in values:
            # haven't seen this value before
            values.append(value)
            counts.append(1)
        elif value in values:
                index = values.index(value)
                counts[index] += 1

    return values, counts 

def get_aus_frequencies(MyPyTable, col_name):
    """Gets the frequency and count of a column by name

    Args:
        MyPyTable(MyPyTable): self of MyPyTable
        col_name(str): name of the column

    Returns:
        values, counts (string, int): name of value and its frequency"""

    rain_col = MyPyTable.get_column(col_name)
    row_index_to_drop = []
    print("range:", len(rain_col), len(MyPyTable.data))
    for i in range(len(rain_col)):
        if rain_col[i] == "No":
            row_index_to_drop.append(i)
    
    count = 0
    row_to_drop = []
    for i in range(len(MyPyTable.data)):
        if i in row_index_to_drop:
            row_to_drop.append(MyPyTable.data[i])

    MyPyTable.drop_rows(row_to_drop)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_col = MyPyTable.get_column('Date')
    yes_col = []
    for month in months:
        yes = 0
        for i in range(len(month_col)):
            if month in month_col[i]:
                yes = yes + 1
        yes_col.append(yes)

    return months, yes_col

def get_sea_frequencies(MyPyTable, col_name):
    """Gets the frequency and count of a column by name

    Args:
        MyPyTable(MyPyTable): self of MyPyTable
        col_name(str): name of the column

    Returns:
        values, counts (string, int): name of value and its frequency"""

    rain_col = MyPyTable.get_column(col_name)
    row_index_to_drop = []
    print("range:", len(rain_col), len(MyPyTable.data))
    for i in range(len(rain_col)):
        if rain_col[i] == "FALSE":
            row_index_to_drop.append(i)
    
    count = 0
    row_to_drop = []
    for i in range(len(MyPyTable.data)):
        if i in row_index_to_drop:
            row_to_drop.append(MyPyTable.data[i])

    MyPyTable.drop_rows(row_to_drop)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_col = MyPyTable.get_column('DATE')
    yes_col = []
    for month in months:
        yes = 0
        for i in range(len(month_col)):
            if month in month_col[i]:
                yes = yes + 1
        yes_col.append(yes)

    return months, yes_col

def get_mpg_frequencies(MyPyTable, col_name):
    """Gets the frequency and count of a column by name

    Args:
        MyPyTable(MyPyTable): self of MyPyTable
        col_name(str): name of the column

    Returns:
        values, counts (string, int): name of value and its frequency"""

    col = MyPyTable.get_column(col_name)

    values = []
    counts = []

    for value in col:
        if value not in values:
            # haven't seen this value before
            if value >= 13 and value < 14:
                values, counts = mpg_val_check(1, values, counts, value)
            elif value == 14:
                values, counts = mpg_val_check(2, values, counts, value)
            elif value > 14 and value <= 16:
                values, counts = mpg_val_check(3, values, counts, value)
            elif value > 16 and value <= 19:
                values, counts = mpg_val_check(4, values, counts, value)
            elif value > 19 and value <= 23:
                values, counts = mpg_val_check(5, values, counts, value)
            elif value > 23 and value <= 26:
                values, counts = mpg_val_check(6, values, counts, value)
            elif value > 26 and value <= 30:
                values, counts = mpg_val_check(7, values, counts, value)
            elif value > 30 and value <= 36:
                values, counts = mpg_val_check(8, values, counts, value)
            elif value > 36 and value <= 44:
                values, counts = mpg_val_check(9, values, counts, value)
            elif value >= 45:
                values, counts = mpg_val_check(10, values, counts, value)
        
    temp_counts = copy.deepcopy(counts)

    #re-order/sort values and temp_counts
    for i in range(len(values)):
        index = values[i]
        temp_counts[index - 1] = counts[i]
    values.sort()
    counts = temp_counts

    return values, counts 

def compute_equal_width_cutoffs(values, num_bins):
    """Gets the equal width of a values list with it's number of bins and returns the positions
        as a list called cutoffs

    Args:
        values(list): values list
        num_bin(int): number of bins

    Returns:
        cutoffs(list): list of positions of equal width for bins"""
    values_range = max(values) - min(values)
    bin_width = values_range / num_bins 
    cutoffs = []
    cutoffs.append(min(values))
    for i in range(num_bins):
        cutoffs.append(cutoffs[i] + bin_width)

    cutoffs = [round(cutoff, 2) for cutoff in cutoffs]
    return cutoffs 

def compute_bin_frequencies(values, cutoffs):
    """Computes bin frequencies/number of occurences

    Args:
        values(list): values list
        cutoff(list): list of positions of equal width for bins

    Returns:
        freq(list): list of frequencies for each cutoff index"""
    freqs = [0 for _ in range(len(cutoffs) - 1)]

    for val in values:
        if val == max(values):
            freqs[-1] += 1
        else:
            for i in range(len(cutoffs) - 1):
                if cutoffs[i] <= val < cutoffs[i + 1]:
                    freqs[i] += 1

    return freqs

def mean(num):
    """Computes mean of list of numbers

    Args:
        num(list): numbers list

    Returns:
        sum(num)/len(num)): mean value"""

    return sum(num)/len(num)

def compute_slope_intercept(x, y):
    """Computes slope intercept

    Args:
        x(list): x values
        y(list): y values

    Returns:
        m(float): m in y = mx + b line equation. Slope
        b(float): b in y = mx + b line equation. Line intercept"""
    mean_x = mean(x)
    mean_y = mean(y) 
    m = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) \
        / sum([(x[i] - mean_x) ** 2 for i in range(len(x))])
    # y = mx + b => y - mx
    b = mean_y - m * mean_x
    return m, b 

def correlation_coeficient(x, y):
    """Computes correlation coeficient

    Args:
        x(list): x values
        y(list): y values

    Returns:
        coeficient(float): coeficient value"""
    n = len(x)
    x_med = sum(x)/len(x)
    y_med = sum(y)/len(y)
    
    numerator = sum([xi*yi for xi,yi in zip(x, y)]) - n * x_med * y_med
    denominator = math.sqrt((sum([xi**2 for xi in x]) - n * x_med**2)*(sum([yi**2 for yi in y]) - n * y_med**2))

    coeficient = round((numerator/denominator), 2)
    return coeficient

def covariance(x, y):
    """Computes covariance

    Args:
        x(list): x values
        y(list): y values

    Returns:
        cov(float): covariance value"""
    n = len(x)
    x_med = sum(x)/len(x)
    y_med = sum(y)/len(y)

    numerator = sum([xi*yi for xi,yi in zip(x, y)]) - n * x_med * y_med
    denominator = n
    
    cov = round((numerator/denominator), 2)
    return cov

def get_occurences_given_columns(table, column_names):
    """Gets the occurence from each column in a given columns list.

    Args:
        table(MyPyTable): given object of MyPyTable
        column_names(list): List of string column names

    Returns:
        count(list): list of frequencies in each correct index matching with the given columns list"""
    column = []
    count = []
    for i in range(len(column_names)):
        count.append(0)
    
    for col in column_names:
        attributes = MyPyTable.get_column(table, col, False)
        column.append(attributes)
    for i in range(len(column)):
        for j in column[i]:
            if j == 1.0:
                count[i] = count[i] + 1
    
    return count

def percentages_columns(table, column_names):
    """Gives the percentage of each column's frequency divided by total column length

    Args:
        table(MyPyTable): given object of MyPyTable
        column_names(list): List of string column names

    Returns:
        percentages(list): list of percentages in each correct index matching with the given columns list"""
    counts = get_occurences_given_columns(table, column_names)
    percentages = []
    col = MyPyTable.get_column(table, column_names[0], False)
    length = len(col)
    for count in counts:
        percentages.append(round((count/length)* 100, 0))
    return percentages

def combine_two_columns(column_names, col1, col2):
    """Creates a MyPyTable from two columns and their column names

    Args:
        column_names(list): List of string column names
        col1(list): List of values from first column
        col2(list): List of values from second column

    Returns:
        table(MyPyTable): Returned MyPyTable with two columns"""
    data = []
    for i in range(len(col1)):
        data.append([col1[i], col2[i]])
      
    table = MyPyTable(column_names, data)
    return table

def convert_attributes(table):
    """Converts IMDb to double digit float and Rotten Tomatoes to string without %

    Args:
        table(MyPyTable): given object of MyPyTable

    Returns:
        imbd_col(list): IMDb list in double digits
        rotten_col(list): Rotten Tomatoes list stripped of %"""
    #IMDb conversion
    col = MyPyTable.get_column(table,'IMDb', False)
    rotten_col = MyPyTable.get_column(table, 'Rotten Tomatoes', False)
    imbd_col = []
    for i in col:
        i = i * 10
        imbd_col.append(i)
    
    #rotten tomatoes conversion
    for a, x in enumerate(rotten_col):
        rotten_col[a] = float(x[:-1])
    
    return imbd_col, rotten_col

def get_ratings_genre(table, genre, rating):
    """Get list with ratings attached with given genre column

    Args:
        table(MyPyTable): given object of MyPyTable
        genre(string): Genre to search for in table
        rating(string): Service provider to pull from in get column

    Returns:
        list(list): list with ratings from each correctly found genre"""
    
    genre_col = MyPyTable.get_column(table, 'Genres', True)

    col = MyPyTable.get_column(table, rating, True)
    list = []
    for i in range(len(genre_col)):
        if genre in genre_col[i]:
            if rating == 'Rotten Tomatoes' and '%' in col[i]:
                col[i] = float(col[i].strip('%'))
            list.append(col[i])
    
    copy_list = copy.deepcopy(list)
    
    for value in list:
        if value == '':
            copy_list.remove(value)

    list = copy_list

    return list

def unique_genres(table):
    """Get list of unique genres within a table

    Args:
        table(MyPyTable): given object of MyPyTable

    Returns:
        values(list): list with unique genres"""
    genre_str = ''
    genre_col = MyPyTable.get_column(table, 'Genres', False)
    vals, counts = get_frequencies(table, 'Genres')
    for v in vals:
        genre_str = genre_str + v + ','
    genre_array = genre_str.split(',')
    
    values = []

    for value in genre_array:
        if value != '':
            if value not in values:
                # haven't seen this value before
                values.append(value)
            elif value in values:
                pass
    return values

def create_list(table, unique_genres, rating):
    """Get list with ratings attached with given genre column

    Args:
        table(MyPyTable): given object of MyPyTable
        unique_genres(list): list of unique genres
        rating(string): Service provider to pull from in get column

    Returns:
        plot_data(list): 2D list with frequencies from each genre in same index values as unique_genres"""
    plot_data = []
    dictionary = {}

    for val in unique_genres:
        dictionary[val] = []
    
    for key in dictionary:
        ratings = get_ratings_genre(table, key, rating)
        dictionary[key] = ratings
    
    for key in dictionary:
        plot_data.append(dictionary[key])


    return plot_data

def get_year_counts(table, platform):
    """Get years of occuring platform game occurences along with their individual frequencies

    Args:
        table(MyPyTable): given object of MyPyTable
        platform(string): platform to search for in table

    Returns:
        values, counts (string, int): name of value and its frequency"
        """
    
    plat_col = MyPyTable.get_column(table, 'Platform', True)
    col = MyPyTable.get_column(table, "Year", True)
    list = []
    for i in range(len(plat_col)):
        if plat_col[i] == platform:
            list.append(col[i])
    
    copy_list = copy.deepcopy(list)
    
    for value in list:
        if value == 'N/A':
            copy_list.remove(value)

    list = copy_list
    list.sort()

    values = []
    counts = []

    for value in list:
        if value not in values:
            # haven't seen this value before
            values.append(value)
            counts.append(1)
        elif value in values:
                index = values.index(value)
                counts[index] += 1

    return values, counts 