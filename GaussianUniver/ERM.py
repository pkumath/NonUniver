# ERM.py

import numpy as np
from scipy.optimize import minimize
from .hermiteFeature import tensorized_hermite_features
from scipy.special import factorial

def generate_data(d, k, alpha):
    """
    Generate dataset with n = alpha * d^k samples.
    Returns:
    - X: array of shape (n, d), standard normal samples.
    - y: array of shape (n,), standard normal samples independent of X.
    """
    n = int(alpha * (d ** k))
    X = np.random.randn(n, d)
    y = np.random.randn(n)
    # y is the first column of X
    # y = X[:, 0]
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    return X, y

def flatten_tensor(T):
    """
    Flatten a symmetric tensor into a vector of unique elements.
    """
    return T.reshape(-1)

def unflatten_tensor(vec, shape):
    """
    Unflatten a vector back into a tensor of given shape.
    """
    return vec.reshape(shape)

def compute_empirical_risk(T_flat, He_flat, y, lambda_reg, loss_function):
    """
    Compute empirical risk.
    """
    n_samples = y.shape[0]
    # Compute predictions
    y_pred = He_flat @ T_flat  # Shape: (n_samples,)
    # Compute loss
    loss_values = loss_function(y, y_pred)
    empirical_risk = (1 / n_samples) * np.sum(loss_values) + lambda_reg * np.dot(T_flat, T_flat)
    return empirical_risk

def compute_empirical_risk_grad(T_flat, He_flat, y, lambda_reg, loss_function_grad):
    """
    Compute gradient of the empirical risk with respect to T.
    """
    n_samples = y.shape[0]
    # Compute predictions
    y_pred = He_flat @ T_flat  # Shape: (n_samples,)
    # Compute gradient of loss
    loss_grad_values = loss_function_grad(y, y_pred)  # Shape: (n_samples,)
    # Compute gradient w.r.t T_flat
    grad = (1 / n_samples) * (He_flat.T @ loss_grad_values) + 2 * lambda_reg * T_flat  # Shape: (d^k,)
    return grad

def empirical_risk_minimization(X, y, k, lambda_reg, loss_function, loss_function_grad, return_grad=False):
    """
    Perform empirical risk minimization to find optimal T.
    """
    n_samples, d = X.shape
    # Determine tensor shape
    tensor_shape = tuple([d] * k)
    T_size = d ** k
    # Initialize T
    T_init = np.zeros(T_size)
    
    # Compute features once and reuse
    print("Computing tensorized Hermite features...")
    He_features = tensorized_hermite_features(X, k)  # Shape: (n_samples, d, d, ..., d)
    He_flat = He_features.reshape(n_samples, -1)  # Shape: (n_samples, d^k)
    print("Features computed.")

    # Define objective function and gradient
    def objective(T_vec):
        return compute_empirical_risk(T_vec, He_flat, y, lambda_reg, loss_function)
    
    def gradient(T_vec):
        return compute_empirical_risk_grad(T_vec, He_flat, y, lambda_reg, loss_function_grad)
    
    # Optimize using scipy.optimize.minimize
    print("Starting optimization...")
    # Compute and output the initial loss
    initial_loss = objective(T_init)

    result = minimize(
        fun=objective,
        x0=T_init,
        jac=gradient,
        method='L-BFGS-B',
        options={'disp': True}
    )
    print("Initial loss value:", initial_loss)
    print("Optimization finished.")
    print("Final loss value:", result.fun)
    # Get the optimized T
    T_opt = unflatten_tensor(result.x, tensor_shape)
    if return_grad:
        return T_opt, unflatten_tensor((1 / n_samples) * (He_flat.T @ loss_function(y, He_flat@ result.x)), tensor_shape)
    return T_opt

# Example loss functions
def squared_loss(y_true, y_pred):
    return 0.5 * (y_true - y_pred) ** 2

def squared_loss_grad(y_true, y_pred):
    return y_pred - y_true

def logistic_loss(y_true, y_pred):
    # Assuming y_true in {0, 1}
    return np.log(1 + np.exp(-y_true * y_pred))

def logistic_loss_grad(y_true, y_pred):
    # Assuming y_true in {0, 1}
    return -y_true / (1 + np.exp(y_true * y_pred))

# Main function
if __name__ == "__main__":
    # Parameters
    lambda_reg = 0.  # Regularization parameter
    alpha = 1.0        # Ratio parameter
    k = 1              # Degree of Hermite polynomials
    d = 5              # Dimension of input features

    # Generate data
    X, y = generate_data(d, k, alpha)
    # For logistic regression, y should be in {0, 1}
    y_classification = (y > 0).astype(np.float64)  # Convert to binary labels

    # Choose loss function and its gradient
    # For regression
    # loss_function = squared_loss
    # loss_function_grad = squared_loss_grad

    # For logistic regression (classification)
    # Uncomment the following lines to use logistic loss
    loss_function = logistic_loss
    loss_function_grad = logistic_loss_grad
    y = y_classification  # Use binary labels

    # Perform empirical risk minimization
    T_hat = empirical_risk_minimization(X, y, k, lambda_reg, loss_function, loss_function_grad)

    # Output the result
    # print("Optimized T:")
    # print(T_hat)
