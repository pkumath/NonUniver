from GaussianUniverRandomFeature.ERM import generate_data, empirical_risk_minimization, generate_random_weights
# from GaussianUniverRandomFeature.hermiteFeature import vectorized_hermite_features
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

def hermite2_activation(x):
    return (x**2 - 1)/np.sqrt(2)


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

    # # Perform empirical risk minimization
    # ba_opt, W = empirical_risk_minimization(X, y, W, lambda_reg, loss_function, loss_function_grad, scaling_factor=scaling_factor)
    
    # # Output the result
    # # print("Optimized coefficients ba:")
    # print("The infinity norm, 1 norm and 2 norm of ba_opt are", np.linalg.norm(ba_opt, np.inf), np.linalg.norm(ba_opt, 1), np.linalg.norm(ba_opt, 2))
    # B = np.sum([ba_opt[j] * np.outer(W[j], W[j]) for j in range(M)], axis=0)
    # # Draw heatmap of B
    # fig, ax = plt.subplots()
    # im = ax.imshow(B)
    # ax.set_xlabel("Dimension")
    # ax.set_ylabel("Dimension")
    # ax.set_title(f"Heatmap of matrix B. Paramters are lambda={lambda_reg}, alpha={alpha},\n n=M=alpha*d**{k}, and we use loss function {loss_function.__name__}. Signal is {signal}. Scaling factor is {scaling_factor}.")
    # fig.colorbar(im)
    # plt.show()


    repeat = 50
    power = 3
    faaf_norm = dict()
    for d in range(2, repeat):
        signal = True
        # Assume k, alpha, lambda_reg, loss_function, and loss_function_grad are defined elsewhere
        X, y = generate_data(d, k, alpha, signal=signal)
        M = int(alpha * (d ** 2))  # Number of random features
        W = generate_random_weights(M, d)
        
        # Compute matrix F_k
        # F_k will have shape (M, d^k). Each row i is w_i^{\otimes k} flattened.
        F_k = np.zeros((M, d ** power))
        for i in range(M):
            w_i = W[i]  # w_i is a vector in R^d
            # Compute w_i^{\otimes k}
            # Start with w_i and iteratively take Kronecker products k-1 times
            v = w_i.copy()
            for _ in range(power - 1):
                v = np.kron(v, w_i)
            F_k[i, :] = v
        print("Shape of F_k:", F_k.shape)


        # # Apply hermite 2 activation to each entry of W_X
        # W_X = hermite2_activation(W_X)
        # Calculate the sample covariance matrix of W_X
        # sample_cov = np.dot(W_X, W_X.T)/M
        # Calculate the operator norm of F matrix
        faaf_norm[d] = np.linalg.norm(F_k, 2)
        
        
    # For each d in the range, plot the operator norm of the sample covariance matrix
    fig, ax = plt.subplots()
    ax.plot(list(faaf_norm.keys()), list(faaf_norm.values()))
    # Plot y=x for comparison
    ax.plot(range(2, repeat), np.array(range(2, repeat))**(0.25), 'r--')
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Operator norm of sample covariance matrix")
    ax.set_title("Operator norm of sample covariance matrix vs. Dimension")
    plt.show()
    # Save the plot
    fig.savefig("figures/Dec10/OperatorNormSampleCovarianceMatrixVsDimension.png")