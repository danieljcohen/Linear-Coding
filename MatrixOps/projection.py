import numpy as np
from print import printMatrix

def vector_projection(v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Projects vector v onto vector u
    Returns the projection of v onto u
    """
    u_norm = np.dot(u, u)
    if u_norm == 0:
        raise ValueError("Cannot project onto the zero vector")
    projection = (np.dot(v, u) / u_norm) * u
    return projection

def subspace_projection(v: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """
    Projects vector v onto the subspace spanned by the basis vectors
    Returns the projection of v onto the subspace
    """
    basis_transpose = np.transpose(basis)
    gram_matrix = np.dot(basis_transpose, basis)
    if np.linalg.det(gram_matrix) == 0:
        raise ValueError("The basis vectors are not linearly independent")
    coefficients = np.linalg.solve(gram_matrix, np.dot(basis_transpose, v))
    projection = np.dot(basis, coefficients)
    return projection

def print_vector(vector: np.ndarray) -> None:
    """
    Prints a vector in a readable format
    """
    print("[" + ' '.join(f"{num:.2f}" for num in vector) + "]")

if __name__ == '__main__':
    v = np.array([3, 4, 5])
    u = np.array([1, 2, 2])

    print("Vector v:")
    print_vector(v)
    
    print("Vector u:")
    print_vector(u)
    
    proj_v_onto_u = vector_projection(v, u)
    print("\nProjection of v onto u:")
    print_vector(proj_v_onto_u)

    basis = np.array([
        [1, 0, 0],
        [0, 1, 1]
    ]).T

    print("\nBasis for subspace:")
    printMatrix(basis.T)
    
    proj_v_onto_subspace = subspace_projection(v, basis)
    print("\nProjection of v onto the subspace spanned by the basis:")
    print_vector(proj_v_onto_subspace)
