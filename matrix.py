from .MatrixOps.cholesky import *
from .MatrixOps.determinant import *
from .MatrixOps.eig import *
from .MatrixOps.elemRowOps import *
from .MatrixOps.inverse import *
from .MatrixOps.matrixmult import *
from .MatrixOps.palu import *
from .MatrixOps.projection import *
from .MatrixOps.qr import *
from .MatrixOps.rref import *
from .MatrixOps.sgd import *
from .MatrixOps.print import printMatrix


def enterMatrix():
    """
    Allows user input for matrix using #
    of rows and columns
    Returns a matrix
    """
    rows = input("Enter # of rows: ")
    columns = input("Enter # of columns: ")
    try:
        rows = int(rows)
        columns = int(columns)
    except ValueError:
        print("Invalid input, please try again.")
        rows = input("Enter # of rows: ")
        columns = input("Enter # of columns: ")
    i = 0
    matrix = []
    while (i < rows):
        while (True):
            temp = input("Enter in the row in #, #, etc... format: ")
            temp = temp.split(', ')
            currentRow = [int(num) for num in temp]
            if(len(currentRow) == columns):
                matrix.append(currentRow)
                i += 1
                break
            else:
                print("Please enter valid input.")
    return matrix



def matrixmult(matrix1, matrix2):
    """
    Multiplies first matrix by the second
    ORDER MATTERS
    Works for vectors, just add them as a matrix w one column
    """
    matrix1NumRows = len(matrix1)
    maxtrix1NumColumns = len(matrix1[0])
    matrix2NumRows = len(matrix2)
    maxtrix2NumColumns = len(matrix2[0])
    
    if maxtrix1NumColumns != matrix2NumRows:
        print("Invalid sizes to multiply")
    
    newMatrixNumRows = matrix1NumRows
    newMatrixNumColumns = maxtrix2NumColumns

    row = 0
    newMatrix = []
    while row < newMatrixNumRows:
        currRow = []
        col = 0
        while col < newMatrixNumColumns:
            colMatrix2 = [row[col] for row in matrix2]
            currRow.append(innerproduct(matrix1[row], colMatrix2))
            col += 1
        newMatrix.append(currRow)
        row += 1
    return newMatrix


def innerproduct(vector1, vector2):
    """
    Used as a helper function for matrix mult
    Vectors are col/row of a matrix
    """
    i = 0
    innerprod = 0
    while i < len(vector1):
        innerprod += (vector1[i] * vector2[i])
        i += 1
    return innerprod


def transpose(matrix):
    """
    Returns the transpose of a matrix
    """
    numRows = len(matrix)
    numColumns = len(matrix[0])

    newMatrix = []

    col = 0
    while (col < numColumns):
        row = 0
        while (row < numRows):
            currRow = [row[col] for row in matrix]
            row += 1
        newMatrix.append(currRow)
        col += 1
    
    return newMatrix


def gramian(matrix):
    """
    Returns the gramian of a matrix
    """
    return matrixmult(transpose(matrix), matrix)

def identityMatrix(size):
    "Makes a identity matrix of a specified size"

    i = 0
    matrix = []
    while i < size:
        row = [0] * size
        row[i] = 1
        matrix.append(row)
        i += 1
    return matrix



if __name__ == '__main__':
    a = [[1, 0, 0], [1, 1, 1], [1, 3, 9],[1, 4, 6]]
    at = transpose(a)
    print("ata")
    printMatrix(matrixmult(at, a))
    b = [[0, 8, 8, 20]]
    print("atb")
    printMatrix(matrixmult(at, b))

    """matrix = [[1, 0, 2, 1], [1, 2, 5, 3], [7, 3, 6, 2]]
    printMatrix(matrix)
    printMatrix(transpose(matrix))
    printMatrix(gramian(matrix))"""
    
    """matrix1 = [[-2, -1, 1], [2, 0, 0], [-1, 1, -2], [1, 1, 0]]
    matrix2 = [[1, 0], [-1, 2], [0, 1]]
    printMatrix(matrixmult(matrix1, matrix2))"""

    
    """matrix = [[3, -9, 0, 0, 15], [-2, 6, 0, 0, -10], [7, -21, 4, 0, 79], [9, -27, 2, 8, 51]]
    printMatrix(matrix)
    print("\n")
    matrix = GaussJordan(matrix)
    printMatrix(matrix)"""