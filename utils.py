"""
Utility functions for PLS vs Ridge Regression Homework.
Data generation helpers for controlled experimental comparisons.
"""

import numpy as np
from numpy.typing import NDArray


def generate_low_rank_data(
    n_samples: int,
    n_features: int,
    n_latent: int,
    noise_std: float = 0.5,
    seed: int = 42,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Generate regression data where y depends on X through a low-rank latent structure.

    X = T @ P' + E_x
    y = T @ q  + e_y

    where T is (n_samples x n_latent), P is (n_features x n_latent),
    and q is (n_latent,).

    This is the setting where PLS should shine because the true generative
    model has latent components that explain both X and y.

    Parameters
    ----------
    n_samples : int
        Number of observations.
    n_features : int
        Number of predictor variables (p).
    n_latent : int
        Number of true latent components driving both X and y.
    noise_std : float
        Standard deviation of observation noise in both X and y.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    beta_true : ndarray of shape (n_features,)
        The true coefficient vector (for reference / evaluation).
    """
    rng = np.random.default_rng(seed)

    # Latent scores
    T = rng.standard_normal((n_samples, n_latent))

    # Loadings (how latent components map to features)
    P = rng.standard_normal((n_features, n_latent))

    # Regression weights in latent space
    q = rng.standard_normal(n_latent) * 3.0

    # Observed data
    X = T @ P.T + noise_std * rng.standard_normal((n_samples, n_features))
    y = T @ q + noise_std * rng.standard_normal(n_samples)

    # True beta in feature space (for noiseless X): beta = P @ inv(P'P) @ q
    # but since P may not be square, use pseudoinverse
    beta_true = P @ np.linalg.solve(P.T @ P, q)

    return X, y, beta_true


def generate_sparse_data(
    n_samples: int,
    n_features: int,
    n_informative: int,
    signal_strength: float = 3.0,
    noise_std: float = 1.0,
    correlation: float = 0.0,
    seed: int = 42,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Generate regression data with a sparse true coefficient vector.

    Only `n_informative` out of `n_features` coefficients are non-zero.
    Features can optionally be correlated (Toeplitz structure).

    Parameters
    ----------
    n_samples : int
        Number of observations.
    n_features : int
        Total number of features (p).
    n_informative : int
        Number of non-zero coefficients (s).
    signal_strength : float
        Magnitude of non-zero coefficients.
    noise_std : float
        Standard deviation of observation noise.
    correlation : float in [0, 1)
        Pairwise correlation between adjacent features (Toeplitz).
        0 means independent features.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    beta_true : ndarray of shape (n_features,)
    """
    rng = np.random.default_rng(seed)

    # Build covariance matrix (Toeplitz)
    if correlation > 0:
        row = correlation ** np.arange(n_features)
        from scipy.linalg import toeplitz

        Sigma = toeplitz(row)
        L = np.linalg.cholesky(Sigma)
        X = rng.standard_normal((n_samples, n_features)) @ L.T
    else:
        X = rng.standard_normal((n_samples, n_features))

    # Sparse coefficient vector
    beta_true = np.zeros(n_features)
    informative_idx = rng.choice(n_features, size=n_informative, replace=False)
    beta_true[informative_idx] = signal_strength * rng.choice(
        [-1, 1], size=n_informative
    )

    y = X @ beta_true + noise_std * rng.standard_normal(n_samples)

    return X, y, beta_true


def generate_collinear_data(
    n_samples: int,
    n_features: int,
    n_groups: int,
    group_size: int,
    noise_std: float = 0.5,
    seed: int = 42,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Generate data with groups of highly collinear features.

    Within each group, features are copies of a single latent variable
    plus small noise. The response depends on the group-level latent
    variables.

    Parameters
    ----------
    n_samples : int
    n_features : int
        Should be >= n_groups * group_size. Extra features are pure noise.
    n_groups : int
        Number of collinear groups.
    group_size : int
        Number of features per group.
    noise_std : float
    seed : int

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    beta_true : ndarray of shape (n_features,)
    """
    rng = np.random.default_rng(seed)

    X = noise_std * rng.standard_normal((n_samples, n_features))

    beta_true = np.zeros(n_features)
    group_weights = rng.standard_normal(n_groups) * 2.0

    for g in range(n_groups):
        latent = rng.standard_normal(n_samples)
        start = g * group_size
        end = start + group_size
        for j in range(start, min(end, n_features)):
            X[:, j] += latent  # collinear within group
        beta_true[start : min(end, n_features)] = group_weights[g] / group_size

    y = X @ beta_true + noise_std * rng.standard_normal(n_samples)

    return X, y, beta_true


def mse(y_true: NDArray, y_pred: NDArray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def relative_mse(beta_hat: NDArray, beta_true: NDArray) -> float:
    """Relative MSE of coefficient estimates: ||beta_hat - beta||^2 / ||beta||^2."""
    return float(np.sum((beta_hat - beta_true) ** 2) / (np.sum(beta_true**2) + 1e-12))
