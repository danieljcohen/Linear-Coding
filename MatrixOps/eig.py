import numpy as np
from print import printMatrix

def calculate_eigenvalues_and_eigenvectors(matrix: np.ndarray):
    """
    Calculate the eigenvalues and eigenvectors of a square matrix
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvalues, eigenvectors

def printMatrix(matrix: np.ndarray) -> None:
    """
    Prints a matrix in a readable format
    """
    for row in matrix:
        print("[" + ' '.join(f"{num:.2f}" for num in row) + "]")

def print_vector(vector: np.ndarray) -> None:
    """
    Prints a vector in a readable format
    """
    print("[" + ' '.join(f"{num:.2f}" for num in vector) + "]")

if __name__ == '__main__':
    A = np.array([
        [4, -2],
        [1, 1]
    ])
    
    print("Matrix A:")
    printMatrix(A)
    
    eigenvalues, eigenvectors = calculate_eigenvalues_and_eigenvectors(A)
    
    print("\nEigenvalues of A:")
    print_vector(eigenvalues)
    
    print("\nEigenvectors of A:")
    printMatrix(eigenvectors)
