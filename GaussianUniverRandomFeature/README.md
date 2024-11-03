# GaussianUniverRandomFeature Package

The `GaussianUniverRandomFeature` package is designed for simulations and empirical risk minimization (ERM) in the context of random feature models. This package includes methods for generating random features based on Hermite polynomials and optimizing coefficients to minimize empirical risk.

## File Descriptions

### 1. [`ERM.py`](./ERM.py)

Implements empirical risk minimization for random feature models by generating random feature mappings and optimizing coefficients.

#### Key Functions:
- **`generate_data(d, k, alpha)`**: Generates a dataset with sample size determined by `d`, `k`, and `alpha`. Returns a matrix `X` of standard normal samples and labels `y`.
- **`generate_random_weights(M, d)`**: Creates `M` random weight vectors uniformly distributed on the unit sphere, used for random feature generation.
- **`compute_F(W, k)`**: Computes the feature vector `F` for each random weight vector in `W` using degree combinations of Hermite polynomials up to degree `k`.
- **`compute_K(Z, F)`**: Constructs the feature matrix `K` by taking the product of Hermite features `Z` (derived from input data) and `F` (derived from random weights).
- **`compute_empirical_risk(ba, K, y, lambda_reg, loss_function)`**: Computes the empirical risk, incorporating a regularization term for the coefficient vector `ba`.
- **`compute_empirical_risk_grad(...)`**: Calculates the gradient of the empirical risk with respect to `ba`.
- **`empirical_risk_minimization(...)`**: Finds optimal coefficients `ba` by minimizing empirical risk using the L-BFGS-B optimization method.

#### Example Loss Functions:
- **`squared_loss`** and **`logistic_loss`**: Loss functions and their gradients for regression and classification.

### 2. [`hermiteFeature.py`](./hermiteFeature.py)

Defines functions for computing Hermite polynomial features in a vectorized or tensorized manner.

#### Key Functions:
- **`hermite_polynomials(x, k)`**: Computes Hermite polynomials up to degree `k` for each input value in `x`, returning a matrix of polynomial values.
- **`vectorized_hermite_features(X, k)`**: Computes vectorized Hermite polynomial features of degree `k` for each sample in `X`, removing duplicate features due to symmetry.
- **`tensorized_hermite_features(X, k)`**: Constructs tensorized Hermite polynomial features, keeping full tensor symmetry for degree `k`.

## Usage

The primary purpose of `GaussianUniverRandomFeature` is to perform empirical risk minimization for random feature models using Hermite polynomial transformations. The process can be summarized as follows:

1. **Data Generation**:
   - Use `generate_data(d, k, alpha)` in `ERM.py` to create synthetic data (`X`, `y`).

2. **Generate Random Weights**:
   - Generate random weight vectors using `generate_random_weights(M, d)` to create a set of features for the random feature model.

3. **Compute Hermite Features**:
   - Compute Hermite polynomial features for both data and random weights with `vectorized_hermite_features(X, k)` and `compute_F(W, k)`.

4. **Compute Empirical Risk**:
   - Call `empirical_risk_minimization(X, y, W, lambda_reg, loss_function, loss_function_grad)` to find the optimal coefficients `ba` by minimizing the empirical risk. It will also return the random weights `W` in order to calculate `B`.

5. **Example Loss Functions**:
   - You may choose between `squared_loss` and `logistic_loss` (or other custom loss functions) for regression or classification tasks.

This package provides flexible tools for exploring random feature models using Hermite polynomial transformations. For detailed usage, refer to the function docstrings within each file.

### 3.[`RandomFeatureTest.py`](./RandomFeatureTest.py)

This file serves as an example of how to use the `GaussianUniverRandomFeature` package. It demonstrates generating random features, performing empirical risk minimization (ERM) using these features, and analyzing the resulting coefficients. Key functions and steps used in this file are explained below.

#### Key Imports

- **`from GaussianUniverRandomFeature.ERM import generate_data, empirical_risk_minimization, generate_random_weights`**: 
    - Imports `generate_data` to create synthetic data (`X`, `y`).
    - Imports `empirical_risk_minimization` to optimize coefficients for the random feature model.
    - Imports `generate_random_weights` to generate random weight vectors for the feature mapping.

#### Step-by-Step Explanation

1. **Parameter Setup**: Defines key parameters for the experiment:
   - `lambda_reg`: Regularization parameter.
   - `alpha`: Ratio parameter that determines the number of random features.
   - `k`: Degree of Hermite polynomials. We need to fix `k=2` for this code!
   - `d`: Dimension of input features.

2. **Data Generation**:
   - Calls `generate_data(d, k, alpha)` to generate synthetic data with `X` as input samples and `y` as target labels.
   - Converts `y` to binary labels (−1 and 1) for classification. (if needed)

3. **Random Weight Generation**:
   - Computes the number of random features, `M = alpha * d^k`.
   - Calls `generate_random_weights(M, d)` to create a set of `M` random weight vectors `W`, uniformly distributed on the unit sphere.

4. **Loss Function Selection**:
   - Defines various loss functions, such as `exponential_loss`, `logistic_loss`, `squared_loss`, `huber_loss`, `hinge_loss`, and `smooth_l1_loss`, along with their gradients.
   - Selects `squared_loss` and `squared_loss_grad` as the loss function and gradient for this example.

5. **Empirical Risk Minimization**:
   - Calls `empirical_risk_minimization(X, y, W, lambda_reg, loss_function, loss_function_grad)` to minimize empirical risk and return the optimized coefficients `ba_opt`, random weights `W`.
   - Computes and prints the infinity, 1, and 2 norms of `ba_opt` for performance evaluation. It also computes the matrix `B` using the information of `ba_opt` and `W`.

6. **Norm Analysis**:
   - Runs a loop over different dimensions `d` to:
     - Re-generate data and weights.
     - Optimize coefficients using `empirical_risk_minimization`.
     - Compute and store the infinity, 1, and 2 norms of the optimized coefficients.
     - Calculates a matrix `B` as the weighted sum of outer products of random weights, using `ba_opt`, and computes the operator norm of `B`.
   - Stores the results for plotting and analysis.

7. **Plotting Results**:
   - Plots the norms (infinity, 1, and 2) of the optimized coefficients across different dimensions.
   - Plots the operator norm of matrix `B` for each dimension `d`.
   - Saves the plots in the `figures/Nov2/` directory with descriptive filenames.

#### How to Run

To run `RandomFeatureTest.py`:
1. **Set Parameters**: Adjust parameters like `lambda_reg`, `alpha`, `k`, and `d` as needed for your experiment.
2. **Choose Loss Function**: Select an appropriate loss function and gradient from the provided options.
3. **Execute the Script**: Running the script will perform empirical risk minimization for random feature models and output plots that show the effects of dimensionality on the norms of optimized coefficients.

