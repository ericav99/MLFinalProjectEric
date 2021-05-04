import matplotlib.pyplot as plt
from mysklearn.mypytable import MyPyTable
import os
import copy
import math

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

    rain_col = MyPyTable.get_column("RainToday")
    row_index_to_drop = []
    for i in range(len(rain_col)):
        if rain_col[i] == "No":
            row_to_drop.append(i)
    row_to_drop = []
    for i in range(len(MyPyTable.data)):
        if i in row_index_to_drop:
            row_to_drop.append(i)
    
    table = MyPyTable.drop_rows(rows_to_drop)
    table.pretty_print()
    return table
    values = []
    counts = []

    '''for value in col:
        if value not in values:
            # haven't seen this value before
            values.append(value)
            counts.append(1)
        elif value in values:
                index = values.index(value)
                counts[index] += 1

    return values, counts'''

def mpg_val_check(num, values, counts, value):
    """Checks mpg values and sees if they already previously exist

    Args:
        num(int): ranking of mpg
        values(list): values list
        counts(list): counts of each value list
        value(int): int in values list

    Returns:
        values, counts (string, int): name of value and its frequency"""
    entered = False
    if num in values:
        index = values.index(num)
        if entered == True:
            counts.append(1)
        if counts[index] > 0 and num in values:
            counts[index] += 1
    else:
        values.append(num)
        entered = True
        if num in values:
            index = values.index(num)
        if entered == True:
            counts.append(1)
        if counts[index] > 0 and num in values:
            counts[index] += 1
    return values, counts

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
