from typing import List
from print import printMatrix

def minor(matrix: List[List[float]], i: int, j: int) -> List[List[float]]:
    """
    Returns the minor of the matrix after removing the i-th row and j-th column
    """
    return [row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])]

def determinant(matrix: List[List[float]]) -> float:
    """
    Calculates the determinant of a square matrix
    """
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for c in range(n):
        det += ((-1) ** c) * matrix[0][c] * determinant(minor(matrix, 0, c))
    return det


if __name__ == '__main__':
    A = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    
    print("Matrix A:")
    printMatrix(A)
    
    det_A = determinant(A)
    print(f"\nDeterminant of A: {det_A:.2f}")

    B = [
        [3, 8],
        [4, 6]
    ]

    print("\nMatrix B:")
    printMatrix(B)
    
    det_B = determinant(B)
    print(f"\nDeterminant of B: {det_B:.2f}")
