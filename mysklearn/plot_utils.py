import matplotlib.pyplot as plt
from mysklearn.mypytable import MyPyTable
import mysklearn.utils as utils
import os
import copy
import math

def pie_chart(x, y):
    """Creates a pie chart with given data

    Args:
        x(list): Labels for pie chunks
        y(list): Values for pie chart
    
    """
    plt.pie(y, labels=x, autopct="%1.1f%%")
    plt.show()

def bar_graph(values, counts):
    """Creates a bar graph with given data

    Args:
        values(list): Values for bar graph
        counts(list): X Labels for bar graph

    """

    plt.bar(range(len(counts)), counts, .45, align="center") 
    plt.xticks(range(len(counts)), values, rotation=45, horizontalalignment="right")
    plt.grid(True)
    plt.show()

def hist_graph(table, column_name):
    """Creates a histogram graph with given data

    Args:
        table(MyPyTable): given table to perform operation
        column_name(string): column name to get column from for hist graph

    """
    col = MyPyTable.get_column(table,column_name, False)

    plt.hist(col, bins=10)
    plt.show()

def percent_hist_graph(table, column_name):
    """Creates a histogram graph with given data and removes the percent sign from given column_names

    Args:
        table(MyPyTable): given table to perform operation
        column_name(string): column name to get column from for hist graph

    """
    col = MyPyTable.get_column(table,column_name, False)

    for i, x in enumerate(col):
        col[i] = float(x[:-1])

    plt.hist(col, bins=10)
    plt.show()

def scatter_plot(table, x_column_name, y_column_name):
    """Creates a scatter plot with given data

    Args:
        table(MyPyTable): given table to perform operation
        column_name(string): column name to get column from for scatter plot. Column on the x axis
        y_column_name(string): column name to get column from for scatter plot. Column on the y axis
    
    Returns:
        coeficient(float): coeficient value
        cov(float): covariance value
    """
    y_col = MyPyTable.get_column(table, y_column_name, False)
    x_col = MyPyTable.get_column(table,x_column_name, False)

    coeficient = utils.correlation_coeficient(x_col, y_col)
    cov = utils.covariance(x_col, y_col)

    m, b = utils.compute_slope_intercept(x_col, y_col)
    plt.scatter(x_col, y_col)
    plt.plot([min(x_col), max(x_col)], [m * min(x_col) + b, m * max(x_col) + b], c="r", label="corr: " + str(coeficient) + ", cov: " + str(cov));
    plt.legend()
    plt.plot()
    plt.show()

    return coeficient, cov