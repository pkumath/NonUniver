import numpy as np
from scipy.sparse.linalg import LinearOperator, svds

def tensor_operator_norm(T, j):
    """
    Computes the j -> k-j operator norm of a symmetric tensor T.

    Parameters:
    T : numpy.ndarray
        Symmetric tensor of order k and dimension d.
    j : int
        Integer between 1 and k-1 specifying the mode.
        if j = 0, the operator norm is the Frobenius norm.

    Returns:
    float
        The j -> k-j operator norm of tensor T.
    """
    if j == 0:
        return np.linalg.norm(T.ravel(), ord=2)
    k = T.ndim
    d = T.shape[0]

    # Define the LinearOperator without forming the full matrix
    def matvec(x):
        # Reshape x to match the dimensions for contraction
        x_tensor = x.reshape([d]*j)
        # Perform tensor contraction over the last j indices
        contracted = np.tensordot(T, x_tensor, axes=([i + (k - j) for i in range(j)], range(j)))
        return contracted.ravel()

    def rmatvec(y):
        # Reshape y to match the dimensions for contraction
        y_tensor = y.reshape([d]*(k - j))
        # Perform tensor contraction over the first k - j indices
        contracted = np.tensordot(T, y_tensor, axes=(list(range(k - j)), range(k - j)))
        return contracted.ravel()

    n_rows = d**(k - j)
    n_cols = d**j

    # Create a LinearOperator
    A = LinearOperator((n_rows, n_cols), matvec=matvec, rmatvec=rmatvec, dtype=T.dtype)

    # Compute the largest singular value using svds
    # We set which='LM' to get the Largest Magnitude singular value
    s = svds(A, k=1, which='LM', return_singular_vectors=False)

    # The operator norm is the largest singular value
    return s[0]

def tensor_operator_flatten(T,j):
    # Flatten the tensor into a matrix
    # The matrix has shape (d^j, d^(k-j))
    k = T.ndim
    d = T.shape[0]
    return np.reshape(T, (d**j, d**(k-j)))

# Example usage:
if __name__ == "__main__":
    # Create a symmetric tensor of order k and dimension d
    k = 4  # Tensor order
    d = 9  # Dimension
    T = np.random.randn(*([d]*k))
    print(f"Shape of the tensor: {T.shape}")
    # Ensure symmetry
    axes = list(range(k))
    perms = [np.array(p) for p in set(tuple(sorted(p)) for p in np.ndindex(*([k]*k))) if len(set(p)) == k]
    T_sym = sum(np.transpose(T, axes=p) for p in perms) / len(perms)
    T = T_sym

    j = 2  # Mode
    norm = tensor_operator_norm(T, j)
    print(f"The {j} -> {k - j} operator norm of the tensor is: {norm}")

    # Create an 2x2 matrix as an example
    T = np.array([[3.0, 0.], [0., 4.]])
    j = 0
    norm = tensor_operator_norm(T, j)
    print(f"The {j} -> {T.ndim - j} operator norm of the matrix is: {norm}")
