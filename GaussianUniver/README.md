# GaussianUniver Package

This package provides functionality for simulations and empirical risk minimization (ERM) for Hermite kernel feature models, utilizing tensorized Hermite features and symmetric tensors.

## File Descriptions

### 1. [`ERM.py`](./ERM.py)

Implements empirical risk minimization for Hermite kernel feature models. 

#### Key Functions:
- **`generate_data(d, k, alpha)`**: Generates a dataset of standard normal samples with dimensions based on `d`, `k`, and `alpha`.
- **`flatten_tensor(T)`**: Flattens a symmetric tensor to a vector.
- **`unflatten_tensor(vec, shape)`**: Restores a vector to its tensor shape.
- **`compute_empirical_risk(T_flat, He_flat, y, lambda_reg, loss_function)`**: Computes the empirical risk given a flattened tensor and features.
- **`compute_empirical_risk_grad(...)`**: Calculates the gradient of the empirical risk with respect to tensor parameters.
- **`empirical_risk_minimization(...)`**: Optimizes tensor parameters to minimize empirical risk.
- **`empirical_risk_minimization_leave_one_out(...)`**: Similar to `empirical_risk_minimization`, excluding a specific sample.
- **`empirical_risk_minimization_constrained_direction(...)`**: Empirical risk minimization with an added symmetry constraint on the tensor.

#### Example Loss Functions:
- **`squared_loss`** and **`logistic_loss`**: Loss functions and gradients for regression and classification tasks.

### 2. [`ERMleaveOneOut.py`](./ERMleaveOneOut.py)

Extends `ERM.py` to include leave-one-out functionality and symmetry-constrained minimization.

#### Additional Key Functions:
- **`empirical_risk_minimization_leave_one_out(...)`**: Excludes a specific sample during minimization, recalculating features and adjusting predictions.
- **`empirical_risk_minimization_constrained_direction(...)`**: Adds symmetry constraints during optimization, preserving specific tensor contraction properties.

### 3. [`hermiteFeature.py`](./hermiteFeature.py)

Provides functions to compute tensorized Hermite features for input data.

#### Key Functions:
- **`hermite_polynomials(x, k)`**: Computes Hermite polynomials up to degree `k` for each input sample.
- **`tensorized_hermite_features(X, k)`**: Constructs tensorized Hermite features of degree `k` for each sample in `X`, combining univariate polynomials.

### 4. [`Optimization.py`](./Optimization.py)

Implements optimization routines and symmetric tensor operations for Hermite models.

#### Key Classes and Functions:
- **`SymmetricTensor`**: Represents symmetric tensors with methods for index retrieval, inner product, Frobenius norm, and gradient-based updates.
- **`train_model(...)`**: Trains a model using Hermite features across specified degrees.
- **`compute_erm(...)`**: Calculates empirical risk for the model, including regularization.

### 5. [`symmetricTensorNorm.py`](./symmetricTensorNorm.py)

Computes operator norms for symmetric tensors.

#### Key Functions:
- **`tensor_operator_norm(T, j)`**: Computes the `j -> k-j` operator norm for a symmetric tensor `T`.
- **`tensor_operator_flatten(T, j)`**: Flattens the tensor `T` to matrix form based on `j` and `k-j` dimensions.

## Usage Example

### [`ERMcontrol.py`](./ERMcontrol.py)

This file serves as a usage example for the `GaussianUniver` package. It demonstrates how to use the core functions for empirical risk minimization and tensor operator norms with a simulated dataset.

#### Key Imports

- **`from GaussianUniver.ERM import generate_data, empirical_risk_minimization`**: Imports `generate_data` to create synthetic data and `empirical_risk_minimization` to optimize tensor parameters.
- **`from GaussianUniver.ERMleaveOneOut import empirical_risk_minimization_constrained_direction`**: Imports `empirical_risk_minimization_constrained_direction` for constrained optimization with a specified direction.
- **`from GaussianUniver.symmetricTensorNorm import tensor_operator_norm, tensor_operator_flatten`**: Imports `tensor_operator_norm` and `tensor_operator_flatten` for calculating operator norms of tensors.

#### Step-by-Step Explanation

1. **Parameter Setup**: Defines key parameters such as `lambda_reg` (regularization), `alpha` (sample size factor), `k` (Hermite polynomial degree), and `d` (feature dimension).
2. **Data Generation**:
   - Uses `generate_data(d, k, alpha)` to generate a dataset `X` and labels `y`.
3. **Loss Function Selection**:
   - Defines several loss functions, including `exponential_loss`, `logistic_loss`, `squared_loss`, `huber_loss`, `hinge_loss`, and `smooth_l1_loss`, along with their gradients.
   - Uses `smooth_l1_loss` and `smooth_l1_loss_grad` as the selected loss function and gradient for this example.
4. **Empirical Risk Minimization**:
   - Calls `empirical_risk_minimization(X, y, k, lambda_reg, loss_function, loss_function_grad)` to perform the optimization, returning `T_hat` as the optimized tensor.
   - Computes the operator norm of `T_hat` with `tensor_operator_norm(T_hat, 1)`.
5. **Constrained Direction Minimization**:
   - Generates a random normalized vector `u` and calls `empirical_risk_minimization_constrained_direction` to perform minimization with the constraint in direction `u`.
   - Computes norms for comparison:
     - `T_operator_norm[(d,j)]`: Operator norm of the constrained tensor.
     - `true_T_operator_norm[(d,j)]`: Operator norm of the true tensor.
     - `comparision_dependent_grad[(d,j)]`: Operator norm for comparison.
6. **Plotting Results**:
   - Uses `matplotlib` to plot the operator norms across dimensions `d`, providing insights into the effect of dimensions on tensor norms.
   - Saves the plot to `figures/` with a descriptive filename.