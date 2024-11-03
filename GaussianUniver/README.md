# GaussianUniver Package

This package provides functionality for simulations and empirical risk minimization (ERM) for Hermite kernel feature models, utilizing tensorized Hermite features and symmetric tensors.

## File Descriptions

### 1. `ERM.py`

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

### 2. `ERMleaveOneOut.py`

Extends `ERM.py` to include leave-one-out functionality and symmetry-constrained minimization.

#### Additional Key Functions:
- **`empirical_risk_minimization_leave_one_out(...)`**: Excludes a specific sample during minimization, recalculating features and adjusting predictions.
- **`empirical_risk_minimization_constrained_direction(...)`**: Adds symmetry constraints during optimization, preserving specific tensor contraction properties.

### 3. `hermiteFeature.py`

Provides functions to compute tensorized Hermite features for input data.

#### Key Functions:
- **`hermite_polynomials(x, k)`**: Computes Hermite polynomials up to degree `k` for each input sample.
- **`tensorized_hermite_features(X, k)`**: Constructs tensorized Hermite features of degree `k` for each sample in `X`, combining univariate polynomials.

### 4. `Optimization.py`

Implements optimization routines and symmetric tensor operations for Hermite models.

#### Key Classes and Functions:
- **`SymmetricTensor`**: Represents symmetric tensors with methods for index retrieval, inner product, Frobenius norm, and gradient-based updates.
- **`train_model(...)`**: Trains a model using Hermite features across specified degrees.
- **`compute_erm(...)`**: Calculates empirical risk for the model, including regularization.

### 5. `symmetricTensorNorm.py`

Computes operator norms for symmetric tensors.

#### Key Functions:
- **`tensor_operator_norm(T, j)`**: Computes the `j -> k-j` operator norm for a symmetric tensor `T`.
- **`tensor_operator_flatten(T, j)`**: Flattens the tensor `T` to matrix form based on `j` and `k-j` dimensions.

## Usage

Each script can be run independently to test specific functionalities or as part of broader simulations. Use `ERMcontrol.py` for example usage of each function.
