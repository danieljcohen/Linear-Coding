import numpy as np
from print import printMatrix

def cholesky_decomposition(matrix: np.ndarray):
    """
    Perform Cholesky decomposition of a positive-definite matrix
    Returns the lower triangular matrix L such that A = LL^T
    """
    L = np.linalg.cholesky(matrix)
    return L


if __name__ == '__main__':
    A = np.array([
        [25, 15, -5],
        [15, 18,  0],
        [-5,  0, 11]
    ])
    
    print("Matrix A:")
    printMatrix(A)
    
    L = cholesky_decomposition(A)
    
    print("\nMatrix L (Lower triangular matrix):")
    printMatrix(L)
    
    print("\nVerification (L * L.T):")
    printMatrix(L @ L.T)
