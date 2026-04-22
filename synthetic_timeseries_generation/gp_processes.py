"""
Gaussian Process inference and synthetic path generation.

Implements GP regression with Cholesky factorization for efficient sampling.
"""

import numpy as np
from scipy.linalg import cholesky, solve_triangular
import warnings

from .gp_kernels import Kernel, RBFKernel


class GaussianProcess:
    """Gaussian Process for time series regression and generation."""

    def __init__(self, kernel=None, noise_std=1e-6):
        """Initialize GP.

        Args:
            kernel: Kernel object (default: RBFKernel)
            noise_std: Noise standard deviation for numerical stability
        """
        self.kernel = kernel or RBFKernel(variance=1.0, lengthscale=1.0)
        self.noise_std = noise_std
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.L = None
        self._is_fitted = False

    def fit(self, X_train, y_train):
        """Fit GP to training data.

        Args:
            X_train: (n, d) training inputs
            y_train: (n,) training targets
        """
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        if y_train.ndim != 1:
            y_train = y_train.ravel()

        self.X_train = X_train
        self.y_train = y_train - np.mean(y_train)  # Center for stability
        self.y_mean = np.mean(y_train)

        # Compute kernel matrix with noise
        K = self.kernel(self.X_train, self.X_train)
        K += self.noise_std ** 2 * np.eye(len(self.X_train))

        # Cholesky factorization for efficient inference
        try:
            self.L = cholesky(K, lower=True)
            self._is_fitted = True
        except np.linalg.LinAlgError:
            warnings.warn("Cholesky failed; increasing noise_std")
            K += 1e-3 * np.eye(len(self.X_train))
            self.L = cholesky(K, lower=True)
            self._is_fitted = True

    def predict(self, X_test, return_std=True):
        """Predict at test points.

        Args:
            X_test: (m, d) test inputs
            return_std: whether to return predictive std

        Returns:
            mean: (m,) predicted means
            std: (m,) predicted standard deviations (if return_std=True)
        """
        if not self._is_fitted:
            raise RuntimeError("GP not fitted. Call fit() first.")

        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)

        K_s = self.kernel(self.X_train, X_test)
        K_ss = self.kernel(X_test, X_test)

        # Solve L y = K_s^T for alpha
        alpha = solve_triangular(self.L, self.y_train, lower=True)
        mean = K_s.T @ alpha + self.y_mean

        if return_std:
            # Solve L v = K_s for v
            v = solve_triangular(self.L, K_s, lower=True)
            std = np.sqrt(np.clip(np.diag(K_ss) - np.sum(v ** 2, axis=0), 0, None))
            return mean, std
        else:
            return mean

    def sample(self, X_test, n_samples=1, seed=None):
        """Draw samples from GP posterior.

        Args:
            X_test: (m, d) test inputs
            n_samples: number of samples
            seed: random seed

        Returns:
            samples: (m, n_samples) posterior samples
        """
        if not self._is_fitted:
            raise RuntimeError("GP not fitted. Call fit() first.")

        if seed is not None:
            np.random.seed(seed)

        mean, std = self.predict(X_test, return_std=True)

        # For proper posterior sampling, use Cholesky of posterior covariance
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)

        K_s = self.kernel(self.X_train, X_test)
        K_ss = self.kernel(X_test, X_test)

        v = solve_triangular(self.L, K_s, lower=True)
        cov = K_ss - v.T @ v + self.noise_std ** 2 * np.eye(len(X_test))

        # Clamp to avoid numerical issues
        cov = (cov + cov.T) / 2
        cov += 1e-8 * np.eye(cov.shape[0])

        L_cov = cholesky(cov, lower=True)
        z = np.random.standard_normal((len(X_test), n_samples))
        samples = mean[:, np.newaxis] + L_cov @ z

        return samples


def generate_synthetic_paths(
    n_paths=100,
    path_length=500,
    kernel=None,
    seed=None,
    trend_strength=0.1,
    volatility_strength=0.15,
):
    """Generate synthetic price paths using Gaussian Processes.

    Mimics Chronos approach: sample from GP priors with various kernel configurations
    to create diverse realistic-looking time series.

    Args:
        n_paths: number of synthetic paths to generate
        path_length: length of each path
        kernel: Kernel object (default: RBFKernel)
        seed: random seed
        trend_strength: strength of trend component (0-1)
        volatility_strength: strength of volatility modulation (0-1)

    Returns:
        paths: (n_paths, path_length) synthetic price log-returns
        """
    if seed is not None:
        np.random.seed(seed)

    if kernel is None:
        kernel = RBFKernel(variance=1.0, lengthscale=10.0)

    paths = []

    for _ in range(n_paths):
        # Sample from GP prior
        gp = GaussianProcess(kernel=kernel)

        # Create inducing points (sparse approximation)
        n_inducing = max(20, path_length // 10)
        x_inducing = np.linspace(0, path_length - 1, n_inducing)
        y_inducing = np.random.normal(0, 1, size=n_inducing)

        # Fit to random points to get a specific function
        gp.fit(x_inducing, y_inducing)

        # Predict over full path
        x_full = np.arange(path_length)
        gp_samples = gp.sample(x_full.reshape(-1, 1), n_samples=1).ravel()

        # Add trend
        trend = trend_strength * np.linspace(-1, 1, path_length)
        path = gp_samples + trend

        # Add stochastic volatility modulation
        vol_mod = 1 + volatility_strength * np.sin(2 * np.pi * np.arange(path_length) / path_length)
        path *= vol_mod

        # Cumulative sum to get price (log-return -> price)
        price = np.exp(np.cumsum(path * 0.01))  # 1% per step baseline

        # Normalize to start at 100
        price = 100 * price / price[0]

        paths.append(price)

    return np.array(paths)


def generate_synthetic_ohlcv(
    n_paths=100,
    path_length=500,
    kernel=None,
    seed=None,
    volume_volatility=0.2,
):
    """Generate synthetic OHLCV data.

    Args:
        n_paths: number of synthetic symbols/paths
        path_length: number of bars per path
        kernel: GP kernel
        seed: random seed
        volume_volatility: volume volatility coefficient

    Returns:
        data: dict with 'open', 'high', 'low', 'close', 'volume' arrays
              each of shape (n_paths, path_length)
    """
    if seed is not None:
        np.random.seed(seed)

    prices = generate_synthetic_paths(
        n_paths=n_paths,
        path_length=path_length,
        kernel=kernel,
        seed=seed,
    )

    data = {}

    # Derive OHLCV from price paths
    for key in ['open', 'high', 'low', 'close']:
        if key == 'close':
            data['close'] = prices
        elif key == 'open':
            # Open is slightly offset from previous close
            data['open'] = np.column_stack([
                prices[:, 0:1],  # first bar opens at base price
                prices[:, :-1] + np.random.normal(0, 0.1 * prices[:, :-1], (n_paths, path_length - 1))
            ])
        elif key == 'high':
            # High is max of open, close, plus random spike
            opens = data['open']
            closes = prices
            highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, 0.2 * prices, (n_paths, path_length)))
            data['high'] = highs
        elif key == 'low':
            # Low is min of open, close, minus random dip
            opens = data['open']
            closes = prices
            lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, 0.2 * prices, (n_paths, path_length)))
            data['low'] = lows

    # Volume with stochastic dynamics
    volume_mean = 1_000_000
    volume_scales = np.random.lognormal(0, 0.5, n_paths)  # Cross-symbol variance
    volume = volume_mean * volume_scales[:, np.newaxis] * np.exp(
        volume_volatility * np.cumsum(np.random.normal(0, 0.1, (n_paths, path_length)), axis=1)
    )
    data['volume'] = np.maximum(volume, 1000)  # Clip to minimum

    return data
