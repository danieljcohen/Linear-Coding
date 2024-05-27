import numpy as np
from print import printMatrix

def calculate_covariance_matrix(data: np.ndarray) -> np.ndarray:
    """
    Calculates the covariance matrix for the given dataset.
    
    data: Input data, matrix of shape (n_samples, n_features)
    Returns the covariance matrix of shape (n_features, n_features)
    """
    # Subtract the mean of each feature (column-wise mean)
    mean_subtracted_data = data - np.mean(data, axis=0)
    # Calculate the covariance matrix
    covariance_matrix = np.dot(mean_subtracted_data.T, mean_subtracted_data) / (data.shape[0] - 1)
    return covariance_matrix


if __name__ == '__main__':
    # Example data (each row is a sample, each column is a feature)
    data = np.array([
        [4.0, 2.0, 0.6],
        [4.2, 2.1, 0.59],
        [3.9, 2.0, 0.58],
        [4.3, 2.1, 0.62],
        [4.1, 2.2, 0.63]
    ])
    
    print("Input data:")
    printMatrix(data)
    
    covariance_matrix = calculate_covariance_matrix(data)
    
    print("\nCovariance Matrix:")
    printMatrix(covariance_matrix)