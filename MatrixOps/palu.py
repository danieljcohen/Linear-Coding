from typing import List, Tuple
from print import printMatrix


def pivotize(matrix: List[List[float]]) -> List[List[float]]:
    """
    Creates the pivoting matrix for matrix.
    """
    m = len(matrix)
    id_mat = [[float(i == j) for i in range(m)] for j in range(m)]
    for j in range(m):
        row = max(range(j, m), key=lambda i: abs(matrix[i][j]))
        if j != row:
            id_mat[j], id_mat[row] = id_mat[row], id_mat[j]
    return id_mat

def lu_decomposition(matrix: List[List[float]]) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    """
    Performs LU Decomposition with partial pivoting
    Returns P, L, and U matrices such that P*A = L*U
    """
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    P = pivotize(matrix)
    PA = matrixmult(P, matrix)
    
    for j in range(n):
        L[j][j] = 1.0
        for i in range(j + 1):
            s1 = sum(U[k][j] * L[i][k] for k in range(i))
            U[i][j] = PA[i][j] - s1
        for i in range(j, n):
            s2 = sum(U[k][j] * L[i][k] for k in range(j))
            L[i][j] = (PA[i][j] - s2) / U[j][j]
    
    return P, L, U

def matrixmult(matrix1: List[List[float]], matrix2: List[List[float]]) -> List[List[float]]:
    """
    Multiplies the first matrix by the second
    ORDER MATTERS
    Works for vectors, just add them as a matrix with one column
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

def innerproduct(vector1: List[float], vector2: List[float]) -> float:
    """
    Used as a helper function for matrix multiplication
    Vectors are col/row of a matrix
    """
    return sum(x * y for x, y in zip(vector1, vector2))

def printMatrix(matrix: List[List[float]]) -> None:
    """
    matrix: The matrix to print
    Prints the matrix in a readable format
    Void function, returns nothing
    """
    for row in matrix:
        print("[" + ' '.join(f"{num:.2f}" for num in row) + "]")

if __name__ == '__main__':
    A = [
        [7, 3, -1, 2],
        [3, 8, 1, -4],
        [-1, 1, 4, -1],
        [2, -4, -1, 6]
    ]
    
    P, L, U = lu_decomposition(A)
    
    print("Matrix A:")
    printMatrix(A)
    
    print("\nPivoting Matrix P:")
    printMatrix(P)
    
    print("\nLower Triangular Matrix L:")
    printMatrix(L)
    
    print("\nUpper Triangular Matrix U:")
    printMatrix(U)
