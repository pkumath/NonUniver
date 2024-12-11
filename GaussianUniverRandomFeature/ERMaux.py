import numpy as np
from scipy.optimize import minimize
from .hermiteFeature import vectorized_hermite_features

def prediction_model(scaling_factor, K, ba, X, centering=False):
    """
    Predict the output of the model for given input X.
    """
    y_pred = scaling_factor * K @ ba
    if centering:
        d = X.shape[1]
        # We first calculate matrix C, where each row is the 2 norm squared of each row of X
        C = np.sum(X ** 2, axis=1)
        # Then we subtract each row of C by d 
        C -= d
        y_pred = y_pred - C/(np.sqrt(2)*d) * (np.sum(ba))
    return y_pred


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

def compute_empirical_risk(ba, K, y, lambda_reg, loss_function, scaling_factor=1, X=1,centering=False):
    """
    Compute empirical risk for given coefficients ba.
    """
    n_samples = y.shape[0]
    M = K.shape[1]
    # y_pred = scaling_factor * K @ ba  # Predictions
    y_pred =  prediction_model(scaling_factor, K, ba, X, centering)
    # y_pred =  K @ ba  # Predictions
    loss_values = loss_function(y, y_pred)
    empirical_risk = (1 / n_samples) * np.sum(loss_values) + lambda_reg * np.dot(ba, ba)
    return empirical_risk

def compute_empirical_risk_grad(ba, K, y, lambda_reg, loss_function_grad, scaling_factor=1, X=1, centering=False):
    """
    Compute gradient of empirical risk with respect to ba, considering the centering term in prediction_model.
    """
    n_samples = y.shape[0]
    M = K.shape[1]
    y_pred = prediction_model(scaling_factor, K, ba, X, centering)
    loss_grad_values = loss_function_grad(y, y_pred)  # Shape: (n_samples,)

    # 基础梯度项（来自 scaling_factor * K @ ba）
    grad = (scaling_factor / n_samples) * (K.T @ loss_grad_values) + 2 * lambda_reg * ba

    # 若 centering 为 True，需要额外考虑 (C/d)*sum(ba) 对梯度的贡献
    if centering:
        d = X.shape[1]
        C = np.sum(X**2, axis=1)  # (n,)
        C -= d
        # 此项对 y_pred 的贡献为 (C/d)*sum(ba)，对 ba 的梯度为 (C/d) 的加权和，再乘以全1向量。
        # 首先计算 (C/d)*loss_grad_values 的点积求和：
        correction = ((C/(np.sqrt(2)*d)) @ loss_grad_values) / n_samples  # 标量

        # sum(ba) 对 ba 的梯度为一个全 1 的向量，因此补偿项是 correction * np.ones(M)
        grad -= correction * np.ones(M)

    return grad

def compute_empirical_risk_hessian(ba, K, y, lambda_reg, loss_function_hess, scaling_factor=1, X=1, centering=False):
    """
    Compute the Hessian of the empirical risk at a given point ba, considering the centering term used in prediction_model.
    
    Parameters:
    - ba: Coefficient vector at which to compute the Hessian. Shape: (M,)
    - K: Feature matrix, shape: (n_samples, M)
    - y: Labels, shape: (n_samples,)
    - lambda_reg: Regularization parameter.
    - loss_function_hess: A function that returns the second derivative of the loss w.r.t. y_pred.
                         Should return an array of shape (n_samples,).
    - scaling_factor: Scaling factor for predictions.
    - X: The input data used for centering if centering=True. Default=1 means no actual centering data.
    - centering: If True, applies the centering correction in the prediction_model.

    Returns:
    - H: Hessian matrix of shape (M, M).
    """
    n_samples = y.shape[0]
    M = K.shape[1]

    # 使用 prediction_model 函数计算 y_pred
    y_pred = prediction_model(scaling_factor, K, ba, X, centering)

    # 计算 loss 的二阶导数值
    loss_hess_values = loss_function_hess(y, y_pred)  # shape: (n_samples,)

    # 根据 prediction_model 的定义推导 Z 矩阵（y_pred对ba的一阶导数）
    # 若 centering=False: y_pred = scaling_factor * K @ ba
    # dy_pred/dba = scaling_factor * K  -> Z = scaling_factor*K
    #
    # 若 centering=True: y_pred = scaling_factor * K @ ba + (C/d)*sum(ba)
    # dy_pred/dba = scaling_factor*K + (C/d)*1_vector
    # 其中 C = np.sum(X**2, axis=1)-d, d = X.shape[1]
    if centering:
        d = X.shape[1]
        C = np.sum(X**2, axis=1)
        C -= d
        Z = scaling_factor * K - (C/(np.sqrt(2)*d))[:, np.newaxis] * np.ones((1, M))
    else:
        Z = scaling_factor * K

    # 计算 Hessian: H = (1/n)*Z^T * diag(loss_hess_values) * Z + 2*lambda_reg*I
    H = (1 / n_samples) * (Z.T @ (loss_hess_values[:, np.newaxis] * Z)) + 2 * lambda_reg * np.eye(M)
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



def empirical_risk_minimization(X_full, y_full, W, lambda_reg, loss_function, loss_function_grad, scaling_factor=1, exclude_index=None, return_FZ=False, return_risk=False, centering=False):
    """
    Perform empirical risk minimization.

    Parameters:
    - X_full: Full dataset features. Shape: (N, d)
    - y_full: Full dataset labels. Shape: (N,)
    - W: Random weights. Shape: (M, d)
    - lambda_reg: Regularization parameter.
    - loss_function: Loss function handle.
    - loss_function_grad: Gradient of the loss function handle.
    - scaling_factor: Scaling factor for the prediction model.
    - exclude_index: If integer, the index of the data point to exclude from optimization.
                     If None or False, include all data points.
    - return_FZ: If True, returns also the matrix K.
    - return_risk: If True, returns also the final risk value.
    - centering: If True, use the centering correction in prediction_model.

    Returns:
    - ba_opt: Optimized coefficients.
    - W: Random weights (unchanged).
    - If return_FZ is True, also returns K.
    - If return_risk is True, also returns the final empirical risk.
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

    # Define objective function and gradient, now passing X and centering
    def objective(ba):
        return compute_empirical_risk(ba, K, y, lambda_reg, loss_function, scaling_factor, X, centering)
    
    def gradient(ba):
        return compute_empirical_risk_grad(ba, K, y, lambda_reg, loss_function_grad, scaling_factor, X, centering)

    # Optimize using scipy.optimize.minimize
    print("Starting optimization...")
    result = minimize(
        fun=objective,
        x0=ba_init,
        jac=gradient,
        method='L-BFGS-B',
        options={'disp': False}
    )
    print("Optimization finished.")
    print("Final loss value:", result.fun)

    # Get the optimized coefficients
    ba_opt = result.x

    if return_FZ and return_risk:
        return ba_opt, W, K, result.fun
    elif return_FZ:
        return ba_opt, W, K
    elif return_risk:
        return ba_opt, W, result.fun

    return ba_opt, W


def compute_S_k(ba, cR_excl_k_at_hat_ba, hat_ba_excl_k, H_excl_k, K_k, y_k, n_full_samples, loss_function, scaling_factor=1, X=1, centering=False):
    # 使用 prediction_model 来计算预测值
    y_pred_k = prediction_model(scaling_factor, K_k, ba, X, centering)
    loss_value_k = loss_function(y_k, y_pred_k)  # 假设返回 (n_samples,) 数组
    diff = ba - hat_ba_excl_k

    # 求和 loss_value_k 再除以 n_full_samples
    S_k = cR_excl_k_at_hat_ba + (1 / n_full_samples) * np.sum(loss_value_k) + 0.5 * diff.T @ H_excl_k @ diff
    return S_k

def compute_S_k_grad(ba, hat_ba_excl_k, H_excl_k, K_k, y_k, n_full_samples, loss_function_grad, scaling_factor=1, X=1, centering=False):
    # 使用 prediction_model 来计算预测值
    y_pred_k = prediction_model(scaling_factor, K_k, ba, X, centering)
    loss_grad_k = loss_function_grad(y_k, y_pred_k)  # (n_samples,)

    # 基础梯度项（对应 scaling_factor * K_k @ ba 部分）
    grad_loss = (scaling_factor / n_full_samples) * (K_k.T @ loss_grad_k)

    # 若 centering 为 True 且 X 可用，则考虑额外的梯度修正
    if centering and not (isinstance(X, int) and X == 1):
        d = X.shape[1]
        C = np.sum(X**2, axis=1) - d
        # 此时需要加上 ((C/d)*sum(ba)) 对梯度的影响：对应对 y_pred 的梯度来说，是一个全1向量乘上 ((C/d)*loss_grad 项的平均)
        correction = ((C/(np.sqrt(2)*d)) @ loss_grad_k) / n_full_samples
        grad_loss -= correction * np.ones_like(ba)

    # 最终梯度为来自损失部分的梯度加上二阶近似项 H_excl_k(ba - hat_ba_excl_k)
    grad_S_k = grad_loss + H_excl_k @ (ba - hat_ba_excl_k)

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
    exclude_index=None,
    return_FZ=False,
    return_risk=False,
    centering=False
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

    def objective_excl_k(ba):
        return compute_empirical_risk(ba, K_excl_k, y_excl_k, lambda_reg, loss_function, scaling_factor, X_excl_k, centering)

    def gradient_excl_k(ba):
        return compute_empirical_risk_grad(ba, K_excl_k, y_excl_k, lambda_reg, loss_function_grad, scaling_factor, X_excl_k, centering)

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
        hat_ba_excl_k, K_excl_k, y_excl_k, lambda_reg, loss_function_hess, scaling_factor, X_excl_k, centering
    )
    eps = 1e-5
    H_excl_k += eps * np.eye(W.shape[0])

    # Define S_k(ba) and its gradient
    print("Defining S_k(ba) and its gradient...")

    def S_k_objective(ba):
        return compute_S_k(
            ba, cR_excl_k_at_hat_ba, hat_ba_excl_k, H_excl_k,
            K_k, y_k, n_full_samples, loss_function, scaling_factor, X_k, centering
        )

    def S_k_gradient(ba):
        return compute_S_k_grad(
            ba, hat_ba_excl_k, H_excl_k, K_k, y_k,
            n_full_samples, loss_function_grad, scaling_factor, X_k, centering
        )

    # Minimize S_k(ba) starting from \hat{\ba}_{\backslash k}
    print("Minimizing S_k(ba) to obtain \\tilde{\\ba}_k(r)...")
    res_tilde = minimize(
        fun=S_k_objective,
        x0=hat_ba_excl_k,
        jac=S_k_gradient,
        method='L-BFGS-B',
        options={'disp': False}
    )
    ba_tilde = res_tilde.x

    if return_FZ and return_risk:
        return ba_tilde, hat_ba_excl_k, K_excl_k, K_k, res_tilde.fun, res.fun
    elif return_FZ:
        return ba_tilde, hat_ba_excl_k, K_excl_k, K_k
    elif return_risk:
        return ba_tilde, hat_ba_excl_k, res_tilde.fun, res.fun

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
