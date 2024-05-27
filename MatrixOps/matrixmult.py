from typing import List
from print import printMatrix


def innerproduct(vector1: List[float], vector2: List[float]) -> float:
    """
    Used as a helper function for matrix multiplication
    Vectors are col/row of a matrix
    """
    return sum(x * y for x, y in zip(vector1, vector2))

def matrixmult(matrix1: List[List[float]], matrix2: List[List[float]]) -> List[List[float]]:
    """
    Multiplies the first matrix by the second
    ORDER MATTERS
    Works for vectors, just add them as a matrix with one column
    
    matrix1: The first matrix
    matrix2: The second matrix
    Returns the result of matrix multiplication
    """
    matrix1NumRows = len(matrix1)
    matrix1NumColumns = len(matrix1[0])
    matrix2NumRows = len(matrix2)
    matrix2NumColumns = len(matrix2[0])
    
    if matrix1NumColumns != matrix2NumRows:
        raise ValueError("Invalid sizes to multiply")

    newMatrix = []
    for row in range(matrix1NumRows):
        currRow = []
        for col in range(matrix2NumColumns):
            colMatrix2 = [matrix2[i][col] for i in range(matrix2NumRows)]
            currRow.append(innerproduct(matrix1[row], colMatrix2))
        newMatrix.append(currRow)
    return newMatrix


def transpose(matrix: List[List[float]]) -> List[List[float]]:
    """
    Returns the transpose of a matrix
    
    matrix: The input matrix
    Returns the transposed matrix
    """
    return [list(row) for row in zip(*matrix)]

if __name__ == '__main__':
    a = [[1, 0, 0], [1, 1, 1], [1, 3, 9], [1, 4, 6]]
    at = transpose(a)
    print("ata")
    printMatrix(matrixmult(at, a))
    
    b = [[0, 8, 8, 20]]
    print("atb")
    printMatrix(matrixmult(at, b))