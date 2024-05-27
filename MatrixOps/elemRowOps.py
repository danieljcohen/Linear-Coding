from print import *

def scaleRow(c, row, matrix):
    """
    c: Constant c to scale by
    row: Row number to scale
    matrix: Current matrix

    Scales a row in a matrix by a constant c
    Returns a matrix
    """
    matrix[row] = [c * num for num in matrix[row]]
    printMatrix(matrix)
    print("\n")
    return matrix


def addRow(row1, row2, c, matrix):
    """
    row1: What will be added
    row2: What you are trying to change
    c: Scalar to multiply row1 by
    matrix: Current matrix
    
    Adds c*row1 to row2
    Returns a matrix
    """
    firstRow = [c*num for num in matrix[row1]]  #Row to be added/scaled
    secondRow = matrix[row2] #Row to be changed
    i = 0
    while i < len(firstRow):
        secondRow[i] += firstRow[i]
        i += 1
    matrix[row2] = secondRow
    printMatrix(matrix)
    print("\n")
    return matrix

def swapRow(row1, row2, matrix):
    """
    Swaps two rows in a matrix
    Returns a matrix
    """
    temp = matrix[row1]
    matrix[row1] = matrix[row2]
    matrix[row2] = temp
    printMatrix(matrix)
    print("\n")
    return matrix