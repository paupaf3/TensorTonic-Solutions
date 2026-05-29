import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    n_rows = len(matrix)
    # Check if the matrix is empty or is not square
    if n_rows == 0 or not all(isinstance(row, list) and len(row) == n_rows for row in matrix):
        return None
        
    # Convert to a numpy array in case a standard Python list is passed
    matrix_array = np.array(matrix)
    
    # Calculate and return the eigenvalues
    eigenvalues = np.linalg.eigvals(matrix_array)
    return eigenvalues
