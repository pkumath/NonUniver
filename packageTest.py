from GaussianUniver.ERM import generate_data, empirical_risk_minimization
from GaussianUniver.symmetricTensorNorm import tensor_operator_norm, tensor_operator_flatten
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



# Main function
if __name__ == "__main__":
    # Parameters
    lambda_reg = 0.2  # Regularization parameter
    alpha = 0.5        # Ratio parameter
    k = 4              # Degree of Hermite polynomials
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
    loss_function = smooth_l1_loss    
    loss_function_grad = smooth_l1_loss_grad
    y = y_classification  # Use binary labels

    # Perform empirical risk minimization
    T_hat = empirical_risk_minimization(X, y, k, lambda_reg, loss_function, loss_function_grad)
    print('Shape of T_hat:', T_hat.shape)
    print('Operator norm of T_hat:', tensor_operator_norm (T_hat, 1))

    repeat = 10
    T_operator_norm = dict()
    # Creat an array to store rank
    T_rank = np.zeros(repeat-2)
    for d in range(2,repeat):
        X, y = generate_data(d, k, alpha)
        y_classification = (y > 0).astype(np.float64)
        y = y_classification
        T_hat = empirical_risk_minimization(X, y, k, lambda_reg, loss_function, loss_function_grad)
        T_hat_matrix = tensor_operator_flatten(T_hat, 1)
        # Rank of T_hat_matrix
        print('Rank of T_hat:', np.linalg.matrix_rank(T_hat_matrix))
        T_rank[d-2] = np.linalg.matrix_rank(T_hat_matrix)
        # print('Frobenuis norm of T_hat:', np.linalg.norm(T_hat_matrix))
        # print('Operator norm of T_hat:', np.linalg.norm(T_hat_matrix, 2))
        # Store all j \to k-j norm
        for j in range(0, k//2+1):
            T_operator_norm[(d,j)] = tensor_operator_norm(T_hat, j)
    print(T_operator_norm)
    # Plot the operator norm
    # For each d and j, plot the operator norm
    
    fig, ax = plt.subplots()
    for j in range(0, k//2+1):
        x = np.arange(2, repeat)
        y = [T_operator_norm[(d,j)] for d in x]
        ax.plot(x, y, label=f"j={j}")
    ax.set_xlabel("Dimension")
    # x axis: d. Begin from 2
    ax.set_xticks(np.arange(2, repeat))
    # Set title
    ax.set_title(f"Operator norm of T_hat with configuration: k={k}, lambda={lambda_reg}, \n alpha={alpha}, loss={loss_function.__name__}")
    ax.set_ylabel("Operator norm")
    ax.legend()
    plt.show()
    # # Save the figure
    # fig.savefig(f"operator_norm_k{k}_lambda{lambda_reg}_alpha{alpha}_loss{loss_function.__name__}_repeat{repeat}.png")
    # import seaborn as sns

    # # 将 T_hat 展开为矩阵
    # T_hat_matrix = tensor_operator_flatten(T_hat, 1)
    # print('Flattened T_hat shape:', T_hat)
    # # Frobenuis norm and operator norm of T_hat
    # print('Frobenuis norm of T_hat:', np.linalg.norm(T_hat_matrix))
    # print('Operator norm of T_hat:', np.linalg.norm(T_hat_matrix, 2))

    # # 使用热力图可视化矩阵
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(T_hat_matrix, cmap='viridis')
    # plt.title('Flattened T_hat')
    # plt.xlabel('Index 1')
    # plt.ylabel('Index 2')
    # plt.show()
    print(T_rank)

    
    