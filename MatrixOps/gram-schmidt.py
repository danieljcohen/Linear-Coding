import numpy as np
from print import printMatrix


def gram_schmidt(matrix: np.ndarray) -> np.ndarray:
    """
    Perform Gram-Schmidt orthogonalization on a set of vectors
    represented by the columns of the input matrix.
    Returns the orthogonalized matrix.
    """
    n, m = matrix.shape
    orthogonal_matrix = np.zeros((n, m))
    for i in range(m):
        # Start with the current vector
        orthogonal_matrix[:, i] = matrix[:, i]
        for j in range(i):
            # Subtract the projection of the current vector onto the previous orthogonal vectors
            proj = np.dot(orthogonal_matrix[:, j], matrix[:, i]) / np.dot(orthogonal_matrix[:, j], orthogonal_matrix[:, j])
            orthogonal_matrix[:, i] -= proj * orthogonal_matrix[:, j]
        # Normalize the orthogonal vector
        orthogonal_matrix[:, i] /= np.linalg.norm(orthogonal_matrix[:, i])
    return orthogonal_matrix


if __name__ == '__main__':
    A = np.array([
        [1, 1, 1],
        [1, 0, 2],
        [1, 2, 0]
    ], dtype=float)
    
    print("Matrix A:")
    printMatrix(A)
    
    Q = gram_schmidt(A)
    
    print("\nOrthogonalized Matrix Q:")
    printMatrix(Q)
    
    print("\nVerification (Q^T * Q):")
    printMatrix(np.dot(Q.T, Q))
