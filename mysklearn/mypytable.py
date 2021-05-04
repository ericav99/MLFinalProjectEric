import mysklearn.myutils as myutils
import copy
import csv 
import statistics
from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.
    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.
        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
         """Prints the table in a nicely formatted grid structure.
         """
         print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).
        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)


    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.
        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.
        Returns:
            tuple of int: rows, cols in the table
        Notes:
            Raise ValueError on invalid col_identifier
        """
        if (isinstance(col_identifier, int) == True):
             col_index = col_identifier
        else:
            try:
                col_index = self.column_names.index(col_identifier)
            except ValueError:
                print("Not an option")
                
        col = []
        if (not include_missing_values):
            for row in self.data:
                if (row[col_index] != "NA"):
                    col.append(row[col_index])
        else:
            for row in self.data:
                col.append(row[col_index])
        return col

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).
        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in range(len(self.data)):
            for point in range(len(self.data[row])):
                try:
                    self.data[row][point] = float(self.data[row][point])
                except ValueError:
                    pass
                    #print(str(self.data[row][point]) + " is not convertible to a float")
        pass

    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.
        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """

        for j in range(len(rows_to_drop)):
            for i in reversed(range(len(self.data))):
                #if (all(map(lambda x, y: x == y, self.data[i], rows_to_drop[j]))):
                if self.data[i] == rows_to_drop[j]:
                    self.data.remove(self.data[i])
                    #del(rows_to_drop[0])
        pass

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.
        Args:
            filename(str): relative path for the CSV file to open and load the contents of.
        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)
        
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        with open(filename, mode='r', encoding='utf-8-sig') as csv_file:
            csv_reader = csv.reader(csv_file)
            row_number = 0
            for row in csv_reader:
                if row_number == 0:
                    self.column_names = row
                else:
                    self.data.append(row)
                row_number += 1
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.
        Args:
            filename(str): relative path for the CSV file to save the contents to.
        Notes:
            Use the csv module.
        """

        with open(filename, mode='w', newline = '') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(self.column_names)
            csv_writer.writerows(self.data)
        pass

    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely baed on key_column_names.
        Args:
            key_column_names(list of str): column names to use as row keys.
        Returns: 
            list of list of obj: list of duplicate rows found
        
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.
        """
        indices = []
        seen = []
        duplicates = []
        for key in key_column_names:
            indices.append(self.column_names.index(key))
        for row in self.data:
            temp = []
            for i in indices:
                temp.append(row[i])
            if temp not in seen:
                seen.append(temp)
            else:
                duplicates.append(row)
        return duplicates

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        for i in reversed(range(len(self.data))):
            for j in range(len(self.data[0])):
                if self.data[i][j] == 'NA':
                    self.data.remove(self.data[i])
                    break
        pass

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.
        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        index = self.column_names.index(col_name)
        relevantColumn = copy.deepcopy(self.get_column(col_name))
        for entry in reversed(range(len(relevantColumn))):
            if relevantColumn[entry] == 'NA':
                del relevantColumn[entry]
        average = sum(relevantColumn)/len(relevantColumn)
        for i in range(len(self.data[0])):
            if self.data[i][index] == 'NA':
                self.data[i][index] = average
        pass

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.
        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        summaryStats = MyPyTable(column_names=["attribute", "min", "max", "mid", "avg", "median"])

        for col in col_names:
            try:
                column = self.get_column(col)
                name = col
                minimum = min(column)
                maximum = max(column)
                median  = statistics.median(column)

                avg = sum(column)/len(column)
                midpoint  = (maximum + minimum) / 2.0
                summaryStats.data.append([name, minimum, maximum, midpoint, avg, median])
            except ValueError:
                print("Not an option")
            except TypeError:
                pass
        return summaryStats

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the inner joined table.
        """
        innerJoined = MyPyTable(column_names = self.column_names + [x for x in other_table.column_names if x not in self.column_names])

        for i in range(len(self.data)):
            for j in range(len(other_table.data)):
                tempOT = []
                tempST = []
                for key in key_column_names:
                    otKey = other_table.column_names.index(key)
                    stKey = self.column_names.index(key)
                    tempOT.append(other_table.data[j][otKey])
                    tempST.append(self.data[i][stKey])
                if tempOT == tempST:
                    innerJoined.data.append(self.data[i] + [x for x in other_table.data[j] if x not in self.data[i]])
        return innerJoined

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the fully outer joined table.
        Notes:
            Pad the attributes with missing values with "NA".
        """
        outerJoined = MyPyTable(column_names = self.column_names + [x for x in other_table.column_names if x not in self.column_names])
        outerJoinTable = []
        temp = []
        for i in range(len(self.data)):
            outerJoinTable.append(['NA' for x in outerJoined.column_names])
            for ii in self.column_names:
                outerJoinTable[i][outerJoined.column_names.index(ii)] = self.data[i][self.column_names.index(ii)]  
            for j in range(len(other_table.data)):
                tempOT = [] #other table
                tempST = [] #self table
                for key in key_column_names:
                    otKey = other_table.column_names.index(key)
                    stKey = self.column_names.index(key)
                    tempOT.append(other_table.data[j][otKey])
                    tempST.append(self.data[i][stKey])
                if tempOT == tempST:
                    temp.append(j)
                    for jj in other_table.column_names:
                        outerJoinTable[i][outerJoined.column_names.index(jj)] = other_table.data[j][other_table.column_names.index(jj)]
        savior = [i for i in range(len(other_table.data)) if i not in temp]
        for other in savior:
            outerJoinTable.append(['NA' for x in outerJoined.column_names])
            for name in other_table.column_names:
                outerJoinTable[len(outerJoinTable)-1][outerJoined.column_names.index(name)] = other_table.data[other][other_table.column_names.index(name)] 
        outerJoined.data = outerJoinTable
        return outerJoined