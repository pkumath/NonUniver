from GaussianUniverRandomFeature.ERMaux import empirical_risk_minimization, generate_data, generate_random_weights, approximate_empirical_risk_minimization
import numpy as np
from matplotlib import pyplot as plt

def exponential_loss(y_true, y_pred):
    return np.exp(-y_true * y_pred)

def exponential_loss_grad(y_true, y_pred):
    return -y_true * np.exp(-y_true * y_pred)

def logistic_loss(y_true, y_pred):
    # Assuming y_true in {0, 1}
    return np.log(1 + np.exp(-y_true * y_pred))

def logistic_loss_grad(y_true, y_pred):
    # Assuming y_true in {0, 1}
    return -y_true / (1 + np.exp(y_true * y_pred))

def squared_loss(y_true, y_pred):
    return 0.5 * (y_true - y_pred) ** 2

def squared_loss_grad(y_true, y_pred):
    return y_pred - y_true

def huber_loss(y_true, y_pred, delta=1.0):
    diff = y_true - y_pred
    abs_diff = np.abs(diff)
    return np.where(abs_diff <= delta, 0.5 * diff**2, delta * (abs_diff - 0.5 * delta))

def huber_loss_grad(y_true, y_pred, delta=1.0):
    diff = y_pred - y_true
    abs_diff = np.abs(diff)
    return np.where(abs_diff <= delta, diff, delta * np.sign(diff))

def hinge_loss(y_true, y_pred):
    return np.maximum(0, 1 - y_true * y_pred)

def hinge_loss_grad(y_true, y_pred):
    return np.where(1 - y_true * y_pred > 0, -y_true, 0)

def smooth_l1_loss(y_true, y_pred):
    diff = np.abs(y_true - y_pred)
    return np.where(diff < 1, 0.5 * diff**2, diff - 0.5)

def smooth_l1_loss_grad(y_true, y_pred):
    diff = y_pred - y_true
    return np.where(np.abs(diff) < 1, diff, np.sign(diff))

def exponential_loss_hess(y_true, y_pred):
    """
    Second derivative (Hessian) of the exponential loss with respect to y_pred.
    """
    return y_true ** 2 * np.exp(-y_true * y_pred)

def logistic_loss_hess(y_true, y_pred):
    """
    Second derivative (Hessian) of the logistic loss with respect to y_pred.
    Assumes y_true in {-1, 1}.
    """
    exp_term = np.exp(y_true * y_pred)
    denom = (1 + exp_term) ** 2
    return exp_term / denom

def squared_loss_hess(y_true, y_pred):
    """
    Second derivative (Hessian) of the squared loss with respect to y_pred.
    """
    return np.ones_like(y_pred)

def huber_loss_hess(y_true, y_pred, delta=1.0):
    """
    Second derivative (Hessian) of the Huber loss with respect to y_pred.
    """
    diff = y_pred - y_true
    abs_diff = np.abs(diff)
    return np.where(abs_diff <= delta, np.ones_like(y_pred), np.zeros_like(y_pred))

def hinge_loss_hess(y_true, y_pred):
    """
    Second derivative (Hessian) of the hinge loss with respect to y_pred.
    Note: The hinge loss is not twice differentiable; the second derivative
    does not exist at certain points. We return zero almost everywhere.
    """
    # Second derivative is zero almost everywhere
    return np.zeros_like(y_pred)

def smooth_l1_loss_hess(y_true, y_pred):
    """
    Second derivative (Hessian) of the Smooth L1 loss with respect to y_pred.
    """
    diff = y_true - y_pred
    abs_diff = np.abs(diff)
    return np.where(abs_diff < 1, np.ones_like(y_pred), np.zeros_like(y_pred))

# Other loss functions can be defined similarly

# Main function
if __name__ == "__main__":
    # Parameters
    lambda_reg = 0.1  # Regularization parameter
    alpha = 1.0       # Ratio parameter
    k = 2             # Degree of Hermite polynomials
    d_max = 100        # Maximum dimension
    signal = True
    scaling_factor = 1
    k_index = 1       # Index of the data point to exclude
    centering = True

    # 使用 logistic loss 函数
    loss_function = logistic_loss
    loss_function_grad = logistic_loss_grad
    loss_function_hess = logistic_loss_hess

    distance_risk_full_vs_approx = {}

    for d in range(10, d_max):
        print(f"Processing dimension d = {d}")
        # Generate data
        X_full, y_full = generate_data(d, k, alpha, signal=signal)
        # For logistic regression, y should be in {-1, 1}
        y_full = (y_full > 0).astype(np.float64)*2-1  # Convert to binary labels
        n_samples = X_full.shape[0]

        # Generate random weights W
        M = int(alpha * (d ** k))
        W = generate_random_weights(M, d)

        # Compute the full minimizer (including all data points)
        # 这里传入 centering=True
        ba_full, _, K, risk_phi = empirical_risk_minimization(
            X_full, y_full, W, lambda_reg, loss_function, loss_function_grad,
            scaling_factor=scaling_factor, exclude_index=None, return_FZ=True, return_risk=True,
            centering=centering
        )

        # Compute the approximate minimizer using the third optimization problem
        # 同样传入 centering=True
        ba_tilde_k, ba_excl_k, K_excel_k, K_k, risk_psi, risl_excel = approximate_empirical_risk_minimization(
            X_full, y_full, W, lambda_reg, loss_function, loss_function_grad,
            loss_function_hess, scaling_factor=scaling_factor, exclude_index=k_index,
            return_FZ=True, return_risk=True,
            centering=centering
        )

        # 计算风险之间的差距
        dist_risk_full_vs_approx = (np.abs(risk_phi - risk_psi)*n_samples)

        # Store distances
        distance_risk_full_vs_approx[d] = dist_risk_full_vs_approx

    # Plot results
    name = 'Lindeberg'
    fig, ax = plt.subplots()
    ax.plot(list(distance_risk_full_vs_approx.keys()), list(distance_risk_full_vs_approx.values()), label='Risk')
    ax.set_xlabel('Dimension d')
    ax.set_ylabel('Distance')
    ax.set_title(f'Risk distance between full and approximate minimizers ({name}) (centering={centering})')
    plt.show()

    # Save the plot
    fig.savefig(f'figures/Dec11/Risk_distance_full_vs_approx_{name}_centering_{centering}_dmax_{d_max}.png')