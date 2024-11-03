from GaussianUniver.ERM import generate_data, empirical_risk_minimization
from GaussianUniver.symmetricTensorNorm import tensor_operator_norm, tensor_operator_flatten
import numpy as np
from GaussianUniver.hermiteFeature import tensorized_hermite_features
from matplotlib import pyplot as plt

# Main function
if __name__ == "__main__":
    # Parameters
    # lambda_reg = 0.2  # Regularization parameter
    # alpha = 0.5        # Ratio parameter
    # k = 3              # Degree of Hermite polynomials
    # d = 2              # Dimension of input features

    # # Generate data
    # X, y = generate_data(d, k, alpha)
    # # For logistic regression, y should be in {0, 1}
    # y_classification = (y > 0).astype(np.float64)  # Convert to binary labels

    # # Choose loss function and its gradient
    # # For regression
    # # loss_function = squared_loss
    # # loss_function_grad = squared_loss_grad

    # # For logistic regression (classification)
    # # Uncomment the following lines to use logistic loss
    # loss_function = smooth_l1_loss    
    # loss_function_grad = smooth_l1_loss_grad
    # y = y_classification  # Use binary labels

    # # Perform empirical risk minimization
    # T_hat = empirical_risk_minimization(X, y, k, lambda_reg, loss_function, loss_function_grad)
    # print('Shape of T_hat:', T_hat.shape)
    # print('Operator norm of T_hat:', tensor_operator_norm (T_hat, 1))
    k = 3
    alpha = 1.0
    repeat = 20
    T_operator_norm = dict()
    for d in range(2,repeat):
        X = np.random.randn(1, d)
        T_hat = tensorized_hermite_features(X, k)
        # Take the absolute value of T_hat
        # T_hat = np.abs(T_hat)
        # T_hat is the average of the first dimension
        T_hat = np.mean(T_hat, axis=0)
        print('Shape of T_hat:', T_hat.shape)
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
    ax.set_title(f"Operator norm of hermite features of degree {k} with alpha={alpha}")
    ax.set_ylabel("Operator norm")
    ax.legend()
    plt.show()
    # Save the figure at figures/Oct7/
    fig.savefig(f"figures/Oct7/operator_norm_of_hermite_features_k{k}_alpha{alpha}_repeat{repeat}.png")