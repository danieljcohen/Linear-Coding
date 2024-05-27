import numpy as np
from print import printMatrix

def qr_decomposition(matrix: np.ndarray):
    """
    Perform QR decomposition of a square matrix
    Returns matrices Q and R such that A = QR
    """
    Q, R = np.linalg.qr(matrix)
    return Q, R


if __name__ == '__main__':
    A = np.array([
        [12, -51, 4],
        [6, 167, -68],
        [-4, 24, -41]
    ])
    
    print("Matrix A:")
    printMatrix(A)
    
    Q, R = qr_decomposition(A)
    
    print("\nMatrix Q (Orthogonal matrix):")
    printMatrix(Q)
    
    print("\nMatrix R (Upper triangular matrix):")
    printMatrix(R)