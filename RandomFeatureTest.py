from GaussianUniverRandomFeature.ERM import generate_data, empirical_risk_minimization, generate_random_weights
# from GaussianUniver.ERMleaveOneOut import empirical_risk_minimization_constrained_direction
# from GaussianUniver.symmetricTensorNorm import tensor_operator_norm, tensor_operator_flatten
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

# Main function
if __name__ == "__main__":
    # Parameters
    lambda_reg = 0.3  # Regularization parameter
    alpha = 1        # Ratio parameter
    k = 2              # Degree of Hermite polynomials
    d = 60              # Dimension of input features
    signal = True


    # Generate data
    X, y = generate_data(d, k, alpha, signal=signal)
    # For logistic regression, y should be in {0, 1}
    y_classification = (y > 0).astype(np.float64)*2-1  # Convert to binary labels

    # Generate random weights W
    M =  int(alpha * (d ** 2))             # Number of random features
    # scaling_factor = 1/np.sqrt(M)
    scaling_factor = 1
    W = generate_random_weights(M, d)

    # 检查 W 的形状和内容
    print("Shape of W:", W.shape)
    print("W:", W)
    # # Choose loss function and its gradient
    # For regression
    loss_function = squared_loss
    loss_function_grad = squared_loss_grad
    y = y  # Use original y
    
    # For logistic regression (classification)
    # # Uncomment the following lines to use logistic loss
    # loss_function = logistic_loss
    # loss_function_grad = logistic_loss_grad
    # y = y_classification  # Use binary labels

    # Perform empirical risk minimization
    ba_opt, W = empirical_risk_minimization(X, y, W, lambda_reg, loss_function, loss_function_grad, scaling_factor=scaling_factor)
    
    # Output the result
    # print("Optimized coefficients ba:")
    print("The infinity norm, 1 norm and 2 norm of ba_opt are", np.linalg.norm(ba_opt, np.inf), np.linalg.norm(ba_opt, 1), np.linalg.norm(ba_opt, 2))
    B = np.sum([ba_opt[j] * np.outer(W[j], W[j]) for j in range(M)], axis=0)
    # Draw heatmap of B
    fig, ax = plt.subplots()
    im = ax.imshow(B)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Dimension")
    ax.set_title(f"Heatmap of matrix B. Paramters are lambda={lambda_reg}, alpha={alpha},\n n=M=alpha*d**{k}, and we use loss function {loss_function.__name__}. Signal is {signal}. Scaling factor is {scaling_factor}.")
    fig.colorbar(im)
    plt.show()


    repeat = 70
    ba_inf_norm = dict()
    ba_1_norm = dict()
    ba_2_norm = dict()
    ba_1_sum = dict()
    B_op_norm = dict()
    B_frobe_norm = dict()
    B_cross_norm = dict()
    for d in range(2,repeat):
        signal = True
        X, y = generate_data(d, k, alpha, signal=signal)
        M =  int(alpha * (d ** 2))             # Number of random features
        # scaling_factor = 1/np.sqrt(M)   
        W = generate_random_weights(M, d)
        # y_classification = 2*(y > 0).astype(np.float64)-1
        # y = y_classification
        ba_hat, W = empirical_risk_minimization(X, y, W, lambda_reg, loss_function, loss_function_grad, scaling_factor=scaling_factor)
        ba_inf_norm[d] = np.linalg.norm(ba_hat, np.inf)
        ba_1_norm[d] = np.linalg.norm(ba_hat, 1)
        ba_2_norm[d] = np.linalg.norm(ba_hat, 2)
        ba_1_sum[d] = np.sum(ba_hat)
        # Calculate the matrix B. Its definition is that, for each i, B = \sum_{j=1}^M ba_hat[j] * row_j(W) * row_j(W)^T
        B = np.sum([ba_hat[j] * np.outer(W[j], W[j]) for j in range(M)], axis=0)
        #l2 norm of the first row of B, but without the first element
        B_cross_norm[d] = np.linalg.norm(B[0, 1:])
        # Ignore the first row and the first column of B
        B = B[1:, 1:]
        B_op_norm[d] = np.linalg.norm(B, 2)
        B_frobe_norm[d] = np.linalg.norm(B)
        
        
    # For each d and j, plot the norms of ba_hat
    
    fig, ax = plt.subplots()
    ax.plot(list(ba_inf_norm.keys()), list(ba_inf_norm.values()), label='infinity norm')
    ax.plot(list(ba_1_norm.keys()), list(ba_1_norm.values()), label='1 norm')
    ax.plot(list(ba_2_norm.keys()), list(ba_2_norm.values()), label='2 norm')
    ax.plot(list(ba_1_sum.keys()), list(ba_1_sum.values()), label='1 sum')
    # Plot 1/d**0.5 for comparison
    ax.plot(np.arange(2, repeat), 0.5/(np.arange(2, repeat)**(-3/4)), label='1/d**0.5')
    ax.legend()
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Norm")
    ax.set_title(f"Norms of the ERM solution hat a for different dimensions,\n lambda={lambda_reg}, alpha={alpha},and we use loss function {loss_function.__name__}. \n  Signal is {signal}. Scaling factor is {scaling_factor}.")

    plt.show()
    # Save the plot
    fig.savefig(f"figures/Dec3/SignalrandomFeature_lambda{lambda_reg}_alpha{alpha}_loss{loss_function.__name__}_repeat{repeat}_signal{signal}_scaling{scaling_factor}.png")

    # For each d and j, plot the norms of ba_hat
    
    fig, ax = plt.subplots()
    ax.plot(list(ba_inf_norm.keys()), list(ba_inf_norm.values()), label='infinity norm')
    # ax.plot(list(ba_1_norm.keys()), list(ba_1_norm.values()), label='1 norm')
    ax.plot(list(ba_2_norm.keys()), list(ba_2_norm.values()), label='2 norm')
    ax.plot(list(ba_1_sum.keys()), list(ba_1_sum.values()), label='1 sum')
    # Plot 1/d**0.5 for comparison
    ax.plot(np.arange(2, repeat), 0.5/(np.arange(2, repeat)**(0.5)), label='1/d**0.5')
    ax.legend()
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Norm")
    ax.set_title(f"Norms of the ERM solution hat a for different dimensions,\n lambda={lambda_reg}, alpha={alpha},and we use loss function {loss_function.__name__}. \n  Signal is {signal}. Scaling factor is {scaling_factor}.")

    plt.show()
    # Save the plot
    fig.savefig(f"figures/Dec3/ZoomSignalrandomFeature_lambda{lambda_reg}_alpha{alpha}_loss{loss_function.__name__}_repeat{repeat}_signal{signal}_scaling{scaling_factor}.png")

    # For each d, plot the operator norm of B
    fig, ax = plt.subplots()
    ax.plot(list(B_op_norm.keys()), list(B_op_norm.values()), label='operator norm of B')
    # Plot 1/d for comparison
    ax.plot(np.arange(2, repeat), 1/(np.arange(2, repeat))**0.5, label='1/d**0.5')
    ax.legend()
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Operator norm of B")

    ax.set_title(f"Operator norm of the matrix B for different dimensions,\n lambda={lambda_reg}, alpha={alpha},and we use loss function {loss_function.__name__}. \n  Signal is {signal}. Scaling factor is {scaling_factor}.")
    plt.show()

    # Save the plot
    fig.savefig(f"figures/Dec3/randomFeature_lambda{lambda_reg}_alpha{alpha}_loss{loss_function.__name__}_repeat{repeat}_B_Signal{signal}_scaling{scaling_factor}.png")

    # For each d, plot the Frobenius norm of B
    fig, ax = plt.subplots()
    ax.plot(list(B_frobe_norm.keys()), list(B_frobe_norm.values()), label='Frobenius norm of B')
    # Plot 1/d for comparison
    ax.plot(np.arange(2, repeat), 1/(np.arange(2, repeat))**0.5, label='1/d**0.5')
    ax.legend()
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Frobenius norm of B")
    ax.set_title(f"Frobenius norm of the matrix B for different dimensions,\n lambda={lambda_reg}, alpha={alpha},and we use loss function {loss_function.__name__}. \n  Signal is {signal}. Scaling factor is {scaling_factor}.")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(list(B_cross_norm.keys()), list(B_cross_norm.values()), label='Cross norm of B')
    # Plot 1/d for comparison
    ax.plot(np.arange(2, repeat), 0.5/(np.arange(2, repeat))**0.5, label='1/d**0.5')
    ax.legend()
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Cross norm of B")
    ax.set_title(f"Cross norm of the matrix B for different dimensions,\n lambda={lambda_reg}, alpha={alpha},and we use loss function {loss_function.__name__}. \n  Signal is {signal}. Scaling factor is {scaling_factor}.")
    plt.show()
    # Save the plot
    fig.savefig(f"figures/Dec3/randomFeature_lambda{lambda_reg}_alpha{alpha}_loss{loss_function.__name__}_repeat{repeat}_B_cross_Signal{signal}_scaling{scaling_factor}.png")
    

