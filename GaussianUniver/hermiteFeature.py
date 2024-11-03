import numpy as np
from scipy.special import hermitenorm
from itertools import combinations_with_replacement
from math import comb
import math
from itertools import permutations
import functools
from itertools import combinations_with_replacement
import operator

def hermite_polynomials(x, k):
    """
    Compute probabilists' Hermite polynomials up to degree k for input x.
    x: array-like, shape (n_samples,)
    Returns: array, shape (n_samples, k+1)
    """
    n_samples = x.shape[0]
    He = np.zeros((n_samples, k+1))
    He[:, 0] = 1
    if k >= 1:
        He[:, 1] = x
    for n in range(2, k+1):
        # Use recursive relation to avoid recomputing polynomials
        He[:, n] = x * He[:, n-1] - (n-1) * He[:, n-2]
    # Normalize the polynomials
    for n in range(k+1):
        He[:, n] /= np.sqrt(math.factorial(n))
    return He

def tensorized_hermite_features(X, k):
    """
    Compute tensorized Hermite polynomial features of degree k.
    X: array-like, shape (n_samples, d)
    Returns: array, shape (n_samples, d, d, ..., d) with k dimensions
    """
    n_samples, d = X.shape
    # Compute univariate Hermite polynomials up to degree k
    He = np.zeros((n_samples, d, k+1))
    for j in range(d):
        He[:, j, :] = hermite_polynomials(X[:, j], k)
    
    # Generate all possible degree combinations (t1, t2, ..., td) such that sum(t_j) = k
    # This is equivalent to the partitions of k into d non-negative integers

    
    # Generate all degree combinations using integer compositions
    # Since order doesn't matter, we can generate compositions efficiently
    # def generate_degree_combinations(k, d):
    #     if d == 1:
    #         yield (k,)
    #         return
    #     for i in range(k+1):
    #         for tail in generate_degree_combinations(k - i, d - 1):
    #             yield (i,) + tail
    
    # degree_combinations = list(generate_degree_combinations(k, d))
    # print(degree_combinations)
    import itertools

    def generate_degree_combinations(k, d):
        # Start with k 'balls' and d-1 dividers to split them into d groups.
        # We are finding combinations of positions to place the dividers.
        for dividers in itertools.combinations(range(k + d - 1), d - 1):
            # Create a list of size d, initialized to 0.
            partition = [0] * d
            # Calculate the number of 'balls' in each partition.
            prev = -1
            for i, divider in enumerate(dividers):
                partition[i] = divider - prev - 1
                prev = divider
            partition[-1] = k + d - 1 - prev - 1
            yield tuple(partition)

    degree_combinations = list(generate_degree_combinations(k, d))

    
    # # Precompute the multinomial coefficients for each degree combination
    # from math import factorial
    # multinomial_coeffs = []
    # for t in degree_combinations:
    #     coeff = factorial(k)
    #     for ti in t:
    #         coeff //= factorial(ti)
    #     multinomial_coeffs.append(coeff)
    # multinomial_coeffs = np.array(multinomial_coeffs)
    
    # Compute the product of Hermite polynomials for each degree combination
    # Initialize the tensorized features
    tensor_shape = [n_samples] + [d] * k
    T = np.zeros(tensor_shape)
    
    # Compute the indices in the tensor for each degree combination
    # Since the tensor is symmetric, we can generate the indices using multiset permutations

    
    # Precompute the indices and the corresponding products
    for idx_combination, t in enumerate(degree_combinations):
        # Generate all unique permutations (indices) corresponding to t
        indices_list = []
        elements = []
        for dim, count in enumerate(t):
            elements.extend([dim] * count)
        # Generate unique permutations of indices
        indices_set = set(permutations(elements))
        indices_array = np.array(list(indices_set))
        # print(indices_array)
        
        # Compute the product of Hermite polynomials for the current degree combination
        He_prod = np.prod([He[:, j, t_j] for j, t_j in enumerate(t)], axis=0)  # Shape: (n_samples,)
        He_prod = He_prod[:, np.newaxis]  # Shape: (n_samples, 1)
        
        # Assign the computed values to the corresponding positions in the tensor
        for idx in indices_array:
            # Build index tuple for advanced indexing
            index_tuple = (np.arange(n_samples),) + tuple(idx.T)
            T[index_tuple] = He_prod[:, 0]
    
    return T

# Example usage:
n_samples = 3
d = 4
k = 3
X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
T = tensorized_hermite_features(X, k)
print(T.shape)  
print(T)

# print(genereate_degree_combinations(3, 4))
