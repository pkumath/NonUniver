import numpy as np
import itertools
from scipy.special import hermite
from scipy.special import factorial
from collections import defaultdict

# Set random seed for reproducibility
np.random.seed(42)

# Helper function to generate all unique multi-indices for symmetric tensors
def generate_multi_indices(d, l):
    # Generates all combinations of indices for a symmetric tensor of order l in d dimensions
    return list(itertools.combinations_with_replacement(range(d), l))

# Class to handle symmetric tensors
class SymmetricTensor:
    def __init__(self, order, dimension):
        self.order = order  # Order of the tensor (degree of polynomial)
        self.dimension = dimension  # Dimension d
        self.indices = generate_multi_indices(dimension, order)  # Unique indices
        self.num_elements = len(self.indices)
        # Initialize tensor elements with zeros or small random values
        self.elements = np.random.randn(self.num_elements) * 0.01

    def get_index(self, idx_tuple):
        # Map a tuple of indices to the position in the elements array
        # Since the tensor is symmetric, we sort the indices
        idx_tuple = tuple(sorted(idx_tuple))
        try:
            return self.indices.index(idx_tuple)
        except ValueError:
            raise ValueError(f"Index {idx_tuple} not found in tensor indices.")

    def inner_product(self, other_tensor):
        # Compute the inner product between two symmetric tensors
        if self.order != other_tensor.order or self.dimension != other_tensor.dimension:
            raise ValueError("Tensors must have the same order and dimension for inner product.")
        return np.dot(self.elements, other_tensor.elements)

    def frobenius_norm(self):
        # Compute the Frobenius norm of the tensor
        return np.linalg.norm(self.elements)

    def update_elements(self, grad, learning_rate):
        # Update tensor elements using gradient descent
        self.elements -= learning_rate * grad
    
    def compute_norm_j_to_k_minus_j_svd(self, j):
        """
        Compute the norm ||T||_{j -> k-j} by flattening the tensor into a matrix and computing its maximum singular value.

        Parameters:
        - j: int, the number of indices to keep after contraction (0 < j < self.order)

        Returns:
        - norm_value: float, the computed norm value
        """
        import numpy as np

        if j <= 0 or j >= self.order:
            raise ValueError("j must be between 1 and the order of the tensor minus 1.")

        d = self.dimension
        k = self.order

        # Compute the dimensions of the matrix
        row_dim = d ** j
        col_dim = d ** (k - j)

        # Initialize the matrix M
        M = np.zeros((row_dim, col_dim))

        # Map multi-indices to flat indices for rows and columns
        row_shape = [d] * j
        col_shape = [d] * (k - j)

        # Iterate over the unique elements of the tensor
        for idx, idx_tuple in enumerate(self.indices):
            idx_row = idx_tuple[:j]
            idx_col = idx_tuple[j:]

            # Compute the row and column indices
            row_idx = np.ravel_multi_index(idx_row, dims=row_shape)
            col_idx = np.ravel_multi_index(idx_col, dims=col_shape)

            # Accumulate the tensor element into the matrix
            M[row_idx, col_idx] += self.elements[idx]

        # Compute the singular values of M
        try:
            singular_values = np.linalg.svd(M, compute_uv=False)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError("SVD did not converge. The matrix might be too large.")

        # The norm is the maximum singular value
        norm_value = singular_values[0]

        return norm_value

# Function to compute univariate normalized Hermite polynomials
def hermite_normalized(n, x):
    # Normalized Hermite polynomial He_n(x)
    Hn = hermite(n)
    norm_factor = 1.0 / np.sqrt(factorial(n))
    return norm_factor * Hn(x)

# Function to compute multivariate Hermite polynomials as symmetric tensors
def compute_hermite_tensor(x, order):
    # x: Input vector of dimension d
    # order: Degree l of the polynomial
    d = x.shape[0]
    tensor = SymmetricTensor(order, d)
    for idx, idx_tuple in enumerate(tensor.indices):
        # Compute the product of univariate Hermite polynomials
        prod = 1.0
        counts = np.bincount(idx_tuple, minlength=d)
        for i in range(d):
            n_i = counts[i]
            if n_i > 0:
                prod *= hermite_normalized(n_i, x[i])
        tensor.elements[idx] = prod
    return tensor

# Logistic loss function and its gradient
def logistic_loss(y_true, y_pred):
    # y_true: True labels (0 or 1)
    # y_pred: Predicted logits
    logits = y_pred
    loss = np.log(1 + np.exp(- (2 * y_true - 1) * logits))
    return loss

def logistic_loss_grad(y_true, y_pred):
    # Gradient of logistic loss with respect to logits
    exp_term = np.exp(- (2 * y_true - 1) * y_pred)
    grad = - (2 * y_true - 1) * exp_term / (1 + exp_term)
    return grad

def train_model(d, k, n, learning_rate=0.01, epochs=10, lambda_reg=0.01, min_degree=1):
    """
    Train the model using only Hermite features of degree between min_degree and k.

    Parameters:
    - d: int, dimension of input data
    - k: int, maximum degree of Hermite polynomials
    - n: int, number of samples
    - learning_rate: float, learning rate for gradient descent
    - epochs: int, number of training epochs
    - lambda_reg: float, regularization coefficient
    - min_degree: int, minimum degree of Hermite polynomials to include (default is 1)

    Returns:
    - T: list of trained SymmetricTensor objects
    - X: numpy array of input data samples
    - y: numpy array of true labels
    """
    # Initialize ground truth tensors for label generation (using all degrees)
    T_star = [SymmetricTensor(order=l, dimension=d) for l in range(1, k+1)]
    # For simplicity, set ground truth tensors to random values
    for tensor in T_star:
        tensor.elements = np.random.randn(tensor.num_elements)
    
    # Generate input data
    X = np.random.randn(n, d)
    
    # Generate labels
    y = np.zeros(n)
    for i in range(n):
        s_i = 0.0
        for l in range(1, k+1):
            H_l_xi = compute_hermite_tensor(X[i], l)
            s_i += T_star[l-1].inner_product(H_l_xi)
        y[i] = 1 if s_i >= 0 else 0
    
    # Initialize model parameter tensors (only degrees from min_degree to k)
    T = [SymmetricTensor(order=l, dimension=d) for l in range(min_degree, k+1)]
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        total_accuracy = 0.0
        # Initialize gradients for each tensor
        gradients = [np.zeros(tensor.num_elements) for tensor in T]
    
        for i in range(n):
            # Forward pass
            f_xi = 0.0
            H_l_xi_list = []
            for idx, l in enumerate(range(min_degree, k+1)):
                H_l_xi = compute_hermite_tensor(X[i], l)
                H_l_xi_list.append(H_l_xi)
                f_xi += T[idx].inner_product(H_l_xi)
    
            # Compute loss
            loss_i = logistic_loss(y[i], f_xi)
            total_loss += loss_i
    
            # Compute accuracy
            y_pred = 1 if f_xi >= 0 else 0
            total_accuracy += 1 if y_pred == y[i] else 0
    
            # Backward pass
            grad_loss = logistic_loss_grad(y[i], f_xi)
            for idx, l in enumerate(range(min_degree, k+1)):
                scaling_factor = (lambda_reg * (d ** l)) / n
                grad_tensor = grad_loss * H_l_xi_list[idx].elements
                # Add regularization gradient
                grad_tensor += 2 * scaling_factor * T[idx].elements
                # Accumulate gradients
                gradients[idx] += grad_tensor / n
    
        # Update parameters
        for idx in range(len(T)):
            T[idx].update_elements(gradients[idx], learning_rate)
    
        # Compute average loss and accuracy
        avg_loss = total_loss / n
        avg_accuracy = total_accuracy / n * 100
    
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {avg_accuracy:.2f}%")
    
    # Return trained model and data
    return T, X, y


def compute_erm(T, X, y, lambda_reg, min_degree=1):
    """
    Compute the empirical risk (ERM) value for the given model and data.

    Parameters:
    - T: list of SymmetricTensor objects (model parameters)
    - X: numpy array of input data samples, shape (n_samples, d)
    - y: numpy array of true labels, shape (n_samples,)
    - lambda_reg: regularization coefficient (lambda_reg)
    - min_degree: minimum degree of Hermite polynomials included in the model

    Returns:
    - erm_value: scalar, the total ERM value including loss and regularization
    """
    n_samples = X.shape[0]
    d = X.shape[1]
    k = max(tensor.order for tensor in T)
    
    total_loss = 0.0

    for i in range(n_samples):
        x_i = X[i]
        y_i = y[i]
        f_xi = 0.0
        for idx, l in enumerate(range(min_degree, k+1)):
            H_l_xi = compute_hermite_tensor(x_i, l)
            f_xi += T[idx].inner_product(H_l_xi)
        # Compute logistic loss for sample i
        loss_i = logistic_loss(y_i, f_xi)
        total_loss += loss_i

    # Average loss over all samples
    avg_loss = total_loss / n_samples

    # Compute regularization term
    reg_loss = 0.0
    for idx, l in enumerate(range(min_degree, k+1)):
        scaling_factor = (lambda_reg * (d ** l)) / n_samples
        reg_loss += scaling_factor * (T[idx].frobenius_norm() ** 2)

    # Total ERM value
    erm_value = avg_loss + reg_loss

    return erm_value


# Set parameters
d = 4          # Dimension
k = 6          # Maximum degree
n = d**k         # Number of samples
learning_rate = 0.01
epochs = 20
lambda_reg = 0.01
min_degree = k  # Only consider the highest degree Hermite features

# Train the model and get the data
trained_T, X_train, y_train = train_model(d, k, n, learning_rate, epochs, lambda_reg, min_degree=min_degree)

# Compute ERM value on training data
erm_value = compute_erm(trained_T, X_train, y_train, lambda_reg, min_degree=min_degree)
print(f"The ERM value on the training data is: {erm_value:.4f}")

norm_svd = trained_T[-1].compute_norm_j_to_k_minus_j_svd(3)
print(f"Norm computed using SVD: {norm_svd}")
norm_svd = trained_T[-1].compute_norm_j_to_k_minus_j_svd(2)
print(f"Norm computed using SVD: {norm_svd}")
norm_svd = trained_T[-1].compute_norm_j_to_k_minus_j_svd(1)
print(f"Norm computed using SVD: {norm_svd}")