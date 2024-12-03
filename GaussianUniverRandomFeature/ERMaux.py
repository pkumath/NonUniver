import numpy as np
from scipy.optimize import minimize
from .hermiteFeature import vectorized_hermite_features

def generate_data(d, k, alpha, signal=True):
    """
    Generate dataset with n = alpha * d^k samples.
    Returns:
    - X: array of shape (n, d), standard normal samples.
    - y: array of shape (n,), standard normal samples independent of X.
    """
    n = int(alpha * (d ** k))
    X = np.random.randn(n, d)
    # y = np.random.randn(n)
    # y = 1/2**0.5*(X[:, 0]**2 - 1)
    if signal:
        y = 1/2**0.5*(X[:, 0]**2 - 1)
    else:
        y = np.random.randn(n)
    return X, y

def generate_random_weights(M, d):
    """
    Generate M random weight vectors uniformly distributed on the unit sphere.
    """
    W = np.random.randn(M, d)
    W /= np.linalg.norm(W, axis=1, keepdims=True)  # Normalize to unit length
    # print("the norm of each row of W is", np.linalg.norm(W, axis=1))
    return W

def compute_F(W, k=2):
    """
    计算每个随机权重向量 w_j 的特征向量 F。
    基于度数组合计算 W 的单项式。

    参数：
    - W: 数组，形状为 (M, d)
    - k: 单项式的总阶数

    返回：
    - F: 数组，形状为 (M, p)，其中 p 是特征的数量
    """
    import itertools

    def generate_degree_combinations(k, d):
        # Generate all degree combinations (t1, t2, ..., td) such that sum(t_j) = k
        for dividers in itertools.combinations(range(k + d - 1), d - 1):
            partition = [0] * d
            prev = -1
            for i, divider in enumerate(dividers):
                partition[i] = divider - prev - 1
                prev = divider
            partition[-1] = k + d - 1 - prev - 1
            yield tuple(partition)


    M, d = W.shape

    # 生成所有可能的度数组合
    degree_combinations = list(generate_degree_combinations(k, d))

    # 对度数组合进行排序，确保与特征顺序一致
    d_max = d
    degree_combinations.sort(
        key=lambda t: (
            sum(1 for ti in t if ti != 0),
            next((i for i, ti in enumerate(t) if ti != 0), d_max),
            t
        )
    )
    # print("degree_combinations in ERM:", degree_combinations)
    # 计算特征
    F = []
    for t in degree_combinations:
        # 对于每个度数组合，计算 W[:, j] ** t_j 的乘积
        # print('shape of W[:, j]:', W[:, 0].shape)
        # print('shape of t:', len(t))
        # print ('j:', t)
        # # Print all possible monomials
        # print('W[:, j] ** t_j:', [W[:, j] ** t_j if t_j != 0 else 1 for j, t_j in enumerate(t)])
        monomial = np.prod(
            [W[:, j] ** t_j if t_j != 0 else np.ones_like(W[:, j]) for j, t_j in enumerate(t)],
            axis=0
        )
        F.append(monomial)
    F = np.column_stack(F)
    return F


def compute_K(Z, F):
    """
    Compute the matrix K = Z @ F.T
    """
    K = Z @ F.T  # Shape: (n_samples, M)
    return K

def compute_empirical_risk(ba, K, y, lambda_reg, loss_function, scaling_factor=1):
    """
    Compute empirical risk for given coefficients ba.
    """
    n_samples = y.shape[0]
    M = K.shape[1]
    y_pred = scaling_factor * K @ ba  # Predictions
    # y_pred =  K @ ba  # Predictions
    loss_values = loss_function(y, y_pred)
    empirical_risk = (1 / n_samples) * np.sum(loss_values) + lambda_reg * np.dot(ba, ba)
    return empirical_risk

def compute_empirical_risk_grad(ba, K, y, lambda_reg, loss_function_grad, scaling_factor=1):
    """
    Compute gradient of empirical risk with respect to ba.
    """
    n_samples = y.shape[0]
    M = K.shape[1]
    y_pred = scaling_factor * K @ ba
    # y_pred =K @ ba
    loss_grad_values = loss_function_grad(y, y_pred)  # Shape: (n_samples,)
    grad = (1 / n_samples) * (scaling_factor * K.T @ loss_grad_values) + 2 * lambda_reg * ba
    # grad = (1 / n_samples) * ( K.T @ loss_grad_values) + 2 * lambda_reg * ba
    return grad

def compute_empirical_risk_hessian(ba, K, y, lambda_reg, loss_function_hess, scaling_factor=1):
    """
    Compute the Hessian of the empirical risk at a given point ba.

    Parameters:
    - ba: Coefficient vector at which to compute the Hessian.
    - K: Feature matrix.
    - y: Labels.
    - lambda_reg: Regularization parameter.
    - loss_function_hess: Second derivative of the loss function.
    - scaling_factor: Scaling factor for predictions.

    Returns:
    - H: Hessian matrix.
    """
    n_samples = y.shape[0]
    y_pred = scaling_factor * K @ ba  # Shape: (n_samples,)
    loss_hess_values = loss_function_hess(y, y_pred)  # Shape: (n_samples,)

    # Compute Hessian matrix
    H = (1 / n_samples) * (
        scaling_factor ** 2 * K.T @ (loss_hess_values[:, np.newaxis] * K)
    ) + 2 * lambda_reg * np.eye(ba.shape[0])
    return H

# def empirical_risk_minimization(X, y, W, lambda_reg, loss_function, loss_function_grad, scaling_factor=1):
#     """
#     Perform empirical risk minimization to find optimal coefficients ba.
#     """
#     n_samples, d = X.shape
#     M = W.shape[0]
#     k = 2  # Degree of Hermite polynomials

#     # Compute vectorized Hermite features Z and F
#     print("Computing vectorized Hermite features for X...")
#     Z = vectorized_hermite_features(X, k)  # Shape: (n_samples, p)
#     print("Features Z computed. Shape:", Z.shape)
    
#     print("Computing vectorized Hermite features for W...")
#     F = compute_F(W)  # Shape: (M, p)
#     print("Features F computed. Shape:", F.shape)
    
#     # Compute matrix K
#     K = compute_K(Z, F)  # Shape: (n_samples, M)
#     print("Matrix K computed. Shape:", K.shape)
    
#     # Initialize coefficients ba
#     ba_init = np.zeros(M)
    
#     # Define objective function and gradient
#     def objective(ba):
#         return compute_empirical_risk(ba, K, y, lambda_reg, loss_function, scaling_factor)
    
#     def gradient(ba):
#         return compute_empirical_risk_grad(ba, K, y, lambda_reg, loss_function_grad, scaling_factor)
    
#     # Optimize using scipy.optimize.minimize
#     print("Starting optimization...")
#     result = minimize(
#         fun=objective,
#         x0=ba_init,
#         jac=gradient,
#         method='L-BFGS-B',
#         options={'disp': True}
#     )
#     print("Optimization finished.")
#     print("Final loss value:", result.fun)
    
#     # Get the optimized coefficients
#     ba_opt = result.x
#     return ba_opt, W



def empirical_risk_minimization(X_full, y_full, W, lambda_reg, loss_function, loss_function_grad, scaling_factor=1, exclude_index=None):
    """
    Perform empirical risk minimization.
    
    Parameters:
    - X_full: Full dataset features.
    - y_full: Full dataset labels.
    - W: Random weights.
    - lambda_reg: Regularization parameter.
    - loss_function: Loss function.
    - loss_function_grad: Gradient of the loss function.
    - exclude_index: If integer, index of the data point to exclude. If None or False, include all data points.
    
    Returns:
    - ba_opt: Optimized coefficients.
    - W: Random weights (unchanged).
    """
    # Exclude the data point if exclude_index is provided
    if isinstance(exclude_index, int):
        print(f"Excluding data point at index {exclude_index} from the optimization.")
        X = np.delete(X_full, exclude_index, axis=0)
        y = np.delete(y_full, exclude_index, axis=0)
    else:
        X = X_full
        y = y_full

    n_samples, d = X.shape
    M = W.shape[0]
    k_degree = 2

    # Compute vectorized Hermite features Z and F
    print("Computing vectorized Hermite features for X...")
    Z = vectorized_hermite_features(X, k_degree)
    print("Features Z computed. Shape:", Z.shape)

    print("Computing vectorized Hermite features for W...")
    F = compute_F(W, k_degree)
    print("Features F computed. Shape:", F.shape)

    # Compute matrix K
    K = compute_K(Z, F)
    print("Matrix K computed. Shape:", K.shape)

    # Initialize coefficients ba
    ba_init = np.zeros(M)

    # Define objective function and gradient
    def objective(ba):
        return compute_empirical_risk(ba, K, y, lambda_reg, loss_function, scaling_factor)
    
    def gradient(ba):
        return compute_empirical_risk_grad(ba, K, y, lambda_reg, loss_function_grad, scaling_factor)
    # Optimize using scipy.optimize.minimize
    print("Starting optimization...")
    result = minimize(
        fun=objective,
        x0=ba_init,
        jac=gradient,
        method='L-BFGS-B',
        options={'disp': True}
    )
    print("Optimization finished.")
    print("Final loss value:", result.fun)

    # Get the optimized coefficients
    ba_opt = result.x
    return ba_opt, W


def compute_S_k(ba, cR_excl_k_at_hat_ba, hat_ba_excl_k, H_excl_k, K_k, y_k, n_full_samples, loss_function, scaling_factor=1):
    y_pred_k = scaling_factor * K_k @ ba
    loss_value_k = loss_function(y_k, y_pred_k)
    diff = ba - hat_ba_excl_k
    S_k = cR_excl_k_at_hat_ba + (1 / n_full_samples) * loss_value_k + 0.5 * diff.T @ H_excl_k @ diff
    # print(f"Shape of S_k: {S_k.shape}, shape of cR_excl_k_at_hat_ba: {cR_excl_k_at_hat_ba.shape}, shape of loss_value_k: {loss_value_k.shape}, shape of diff: {diff.shape}")
    return S_k

def compute_S_k_grad(ba, hat_ba_excl_k, H_excl_k, K_k, y_k, n_full_samples, loss_function_grad, scaling_factor=1):
    y_pred_k = scaling_factor * K_k @ ba
    loss_grad_k = loss_function_grad(y_k, y_pred_k)
    grad_loss = (1 / n_full_samples) * (scaling_factor * K_k.T * loss_grad_k)
    grad_loss = grad_loss.flatten()
    grad_S_k = grad_loss + H_excl_k @ (ba - hat_ba_excl_k)
    # print(f"Shape of grad_S_k: {grad_S_k.shape}, shape of grad_loss: {grad_loss.shape}, shape of H_excl_k: {H_excl_k.shape}, shape of ba: {ba.shape}, shape of hat_ba_excl_k: {hat_ba_excl_k.shape}")
    return grad_S_k


def approximate_empirical_risk_minimization(
    X_full,
    y_full,
    W,
    lambda_reg,
    loss_function,
    loss_function_grad,
    loss_function_hess,
    scaling_factor=1,
    exclude_index=None
):
    if exclude_index is None:
        raise ValueError("exclude_index must be provided for Ψ_k(r) computation.")

    n_full_samples = X_full.shape[0]

    # Exclude the k-th data point to compute \hat{\ba}_{\backslash k}
    X_excl_k = np.delete(X_full, exclude_index, axis=0)
    y_excl_k = np.delete(y_full, exclude_index, axis=0)

    # Extract the k-th data point
    X_k = X_full[exclude_index:exclude_index+1, :]
    y_k = y_full[exclude_index]

    # Compute features
    k_degree = 2
    Z_excl_k = vectorized_hermite_features(X_excl_k, k_degree)
    Z_k = vectorized_hermite_features(X_k, k_degree)
    F = compute_F(W, k_degree)

    # Compute K matrices
    K_excl_k = compute_K(Z_excl_k, F)
    K_k = compute_K(Z_k, F)

    # Compute \hat{\ba}_{\backslash k}
    print(f"Computing \\hat{{\\ba}}_{{\\backslash k}} by excluding data point at index {exclude_index}...")
    ba_init = np.zeros(W.shape[0])

    # Reuse compute_empirical_risk and compute_empirical_risk_grad
    def objective_excl_k(ba):
        return compute_empirical_risk(ba, K_excl_k, y_excl_k, lambda_reg, loss_function, scaling_factor)

    def gradient_excl_k(ba):
        return compute_empirical_risk_grad(ba, K_excl_k, y_excl_k, lambda_reg, loss_function_grad, scaling_factor)

    res = minimize(
        fun=objective_excl_k,
        x0=ba_init,
        jac=gradient_excl_k,
        method='L-BFGS-B',
        options={'disp': False}
    )
    hat_ba_excl_k = res.x

    # Compute \cR_{\backslash k} at \hat{\ba}_{\backslash k}
    cR_excl_k_at_hat_ba = objective_excl_k(hat_ba_excl_k)

    # Compute Hessian \bH_{\backslash k} at \hat{\ba}_{\backslash k}
    print("Computing Hessian \\bH_{\\backslash k} at \\hat{\\ba}_{\\backslash k}...")
    H_excl_k = compute_empirical_risk_hessian(
        hat_ba_excl_k, K_excl_k, y_excl_k, lambda_reg, loss_function_hess, scaling_factor
    )
    eps = 1e-5
    H_excl_k += eps * np.eye(W.shape[0])

    # Define S_k(ba) and its gradient
    print("Defining S_k(ba) and its gradient...")

    def S_k_objective(ba):
        return compute_S_k(
            ba, cR_excl_k_at_hat_ba, hat_ba_excl_k, H_excl_k,
            K_k, y_k, n_full_samples, loss_function, scaling_factor
        )

    def S_k_gradient(ba):
        return compute_S_k_grad(
            ba, hat_ba_excl_k, H_excl_k, K_k, y_k,
            n_full_samples, loss_function_grad, scaling_factor
        )

    # Minimize S_k(ba) starting from \hat{\ba}_{\backslash k}
    print("Minimizing S_k(ba) to obtain \\tilde{\\ba}_k(r)...")
    res_tilde = minimize(
        fun=S_k_objective,
        x0=hat_ba_excl_k,
        jac=S_k_gradient,
        method='L-BFGS-B',
        options={'disp': True}
    )
    ba_tilde = res_tilde.x

    return ba_tilde, hat_ba_excl_k


# # Include necessary functions from hermiteFeature module
# def hermite_polynomials(x, k):
#     """
#     Compute probabilists' Hermite polynomials up to degree k for input x.
#     x: array-like, shape (n_samples,)
#     Returns: array, shape (n_samples, k+1)
#     """
#     n_samples = x.shape[0]
#     He = np.zeros((n_samples, k+1))
#     He[:, 0] = 1
#     if k >= 1:
#         He[:, 1] = x
#     for n in range(2, k+1):
#         He[:, n] = x * He[:, n-1] - (n-1) * He[:, n-2]
#     # Normalize the polynomials
#     for n in range(k+1):
#         He[:, n] /= np.sqrt(math.factorial(n))
#     return He

# def generate_degree_combinations(k, d):
#     """
#     Generate all possible degree combinations (t1, t2, ..., td) such that sum(t_j) = k
#     """
#     for dividers in itertools.combinations(range(k + d - 1), d - 1):
#         partition = [0] * d
#         prev = -1
#         for i, divider in enumerate(dividers):
#             partition[i] = divider - prev - 1
#             prev = divider
#         partition[-1] = k + d - 1 - prev - 1
#         yield tuple(partition)

# Loss functions
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
    import itertools
    import math

    # Parameters
    lambda_reg = 0.1  # Regularization parameter
    alpha = 1.0        # Ratio parameter
    k = 2              # Degree of Hermite polynomials
    d =10              # Dimension of input features
    M =  int(alpha * (d ** k))             # Number of random features

    # Generate data
    X, y = generate_data(d, k, alpha, signal=True)
    # For logistic regression, y should be in {0, 1}
    y_classification = (y > 0).astype(np.float64)*2-1  # Convert to binary labels

    # Generate random weights W
    W = generate_random_weights(M, d)
    
    # # Choose loss function and its gradient
    # # For regression
    # loss_function = squared_loss
    # loss_function_grad = squared_loss_grad
    # y = y  # Use original y
    
    # For logistic regression (classification)
    # Uncomment the following lines to use logistic loss
    loss_function = logistic_loss
    loss_function_grad = logistic_loss_grad
    y = y_classification  # Use binary labels

    # Perform empirical risk minimization
    ba_opt = empirical_risk_minimization(X, y, W, lambda_reg, loss_function, loss_function_grad)
    
    # Output the result
    # print("Optimized coefficients ba:")
    print("The infinity norm, 1 norm and 2 norm of ba_opt are", np.linalg.norm(ba_opt, np.inf), np.linalg.norm(ba_opt, 1), np.linalg.norm(ba_opt, 2))
