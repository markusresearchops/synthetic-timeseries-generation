"""
Financial price process models: GBM, mean-reverting processes, etc.

Complements GP generation with parametric stochastic process models.
"""

import numpy as np


def generate_geometric_brownian_motion(
    S0=100,
    mu=0.0001,  # drift per step (small for 1-min bars)
    sigma=0.001,  # volatility per step
    n_steps=500,
    n_paths=100,
    seed=None,
):
    """Generate GBM price paths.

    dS = μS dt + σS dW

    Args:
        S0: initial price
        mu: drift coefficient
        sigma: volatility coefficient
        n_steps: number of steps per path
        n_paths: number of paths
        seed: random seed

    Returns:
        paths: (n_paths, n_steps) price arrays
    """
    if seed is not None:
        np.random.seed(seed)

    # Pre-allocate
    paths = np.zeros((n_paths, n_steps))
    paths[:, 0] = S0

    # Generate standard normal increments
    dW = np.random.normal(0, np.sqrt(1.0), (n_paths, n_steps - 1))

    # Euler scheme
    for t in range(n_steps - 1):
        paths[:, t + 1] = paths[:, t] * np.exp(
            (mu - 0.5 * sigma ** 2) + sigma * dW[:, t]
        )

    return paths


def generate_mean_reverting_process(
    S0=100,
    long_mean=100,
    mean_reversion_speed=0.05,
    sigma=0.015,
    n_steps=500,
    n_paths=100,
    seed=None,
):
    """Generate mean-reverting (Ornstein-Uhlenbeck-like) price paths.

    dS = κ(S_long - S) dt + σ dW

    Realistic for short-horizon financial data: prices mean-revert after shocks.

    Args:
        S0: initial price
        long_mean: long-run equilibrium price
        mean_reversion_speed: speed of reversion (kappa)
        sigma: volatility
        n_steps: number of steps
        n_paths: number of paths
        seed: random seed

    Returns:
        paths: (n_paths, n_steps) price arrays
    """
    if seed is not None:
        np.random.seed(seed)

    paths = np.zeros((n_paths, n_steps))
    paths[:, 0] = S0

    dW = np.random.normal(0, np.sqrt(1.0), (n_paths, n_steps - 1))

    for t in range(n_steps - 1):
        drift = mean_reversion_speed * (long_mean - paths[:, t])
        paths[:, t + 1] = paths[:, t] + drift + sigma * dW[:, t]

    return paths


def generate_regime_switching_process(
    S0=100,
    n_steps=500,
    n_paths=100,
    n_regimes=2,
    regime_persistence=0.95,
    seed=None,
):
    """Generate prices with regime-switching dynamics.

    Two regimes: low-vol normal trading, high-vol stress.
    Models the non-stationarity of financial data.

    Args:
        S0: initial price
        n_steps: number of steps
        n_paths: number of paths
        n_regimes: number of regimes
        regime_persistence: probability of staying in same regime
        seed: random seed

    Returns:
        paths: (n_paths, n_steps) price arrays
        regimes: (n_paths, n_steps) regime indicators
    """
    if seed is not None:
        np.random.seed(seed)

    paths = np.zeros((n_paths, n_steps))
    regimes = np.zeros((n_paths, n_steps), dtype=int)
    paths[:, 0] = S0

    # Regime parameters
    regime_mus = np.array([0.0001, -0.0001])  # drift in each regime
    regime_sigmas = np.array([0.001, 0.003])  # vol in each regime

    # Initialize regimes
    regimes[:, 0] = np.random.randint(0, n_regimes, n_paths)

    # Transition matrix
    trans_prob = regime_persistence
    trans_matrix = np.array([
        [trans_prob, 1 - trans_prob],
        [1 - trans_prob, trans_prob]
    ])

    dW = np.random.normal(0, np.sqrt(1.0), (n_paths, n_steps - 1))
    regime_rand = np.random.uniform(0, 1, (n_paths, n_steps - 1))

    for t in range(n_steps - 1):
        # Transition regimes
        for i in range(n_paths):
            current_regime = regimes[i, t]
            next_regime_prob = trans_matrix[current_regime, :]
            if regime_rand[i, t] < next_regime_prob[1]:
                regimes[i, t + 1] = 1
            else:
                regimes[i, t + 1] = 0

        # Step prices using regime parameters
        current_regimes = regimes[:, t]
        mu = regime_mus[current_regimes]
        sigma = regime_sigmas[current_regimes]

        paths[:, t + 1] = paths[:, t] * np.exp(
            (mu - 0.5 * sigma ** 2) + sigma * dW[:, t]
        )

    return paths, regimes


def generate_stochastic_volatility_process(
    S0=100,
    n_steps=500,
    n_paths=100,
    vol_mean=0.001,
    vol_speed=0.1,
    vol_vol=0.0005,
    seed=None,
):
    """Generate prices with stochastic volatility (Heston-like).

    dS = r * S dt + sqrt(v) * S dW_S
    dv = κ(θ - v) dt + σ_v * sqrt(v) dW_v

    Models volatility clustering observed in real financial data.

    Args:
        S0: initial price
        n_steps: number of steps
        n_paths: number of paths
        vol_mean: long-run mean volatility
        vol_speed: mean-reversion speed for volatility
        vol_vol: volatility of volatility
        seed: random seed

    Returns:
        paths: (n_paths, n_steps) price arrays
        volatilities: (n_paths, n_steps) instantaneous volatility
    """
    if seed is not None:
        np.random.seed(seed)

    paths = np.zeros((n_paths, n_steps))
    vols = np.zeros((n_paths, n_steps))
    paths[:, 0] = S0
    vols[:, 0] = vol_mean

    # Correlated Brownian increments
    dW_s = np.random.normal(0, np.sqrt(1.0), (n_paths, n_steps - 1))
    dW_v = np.random.normal(0, np.sqrt(1.0), (n_paths, n_steps - 1))

    # Correlation between price and vol (typically negative for equities)
    rho = -0.7
    dW_s_adj = rho * dW_v + np.sqrt(1 - rho ** 2) * dW_s

    for t in range(n_steps - 1):
        # Vol update (ensure positivity)
        vol_drift = vol_speed * (vol_mean - vols[:, t])
        vols[:, t + 1] = np.abs(vols[:, t] + vol_drift + vol_vol * np.sqrt(vols[:, t]) * dW_v[:, t])

        # Price update
        paths[:, t + 1] = paths[:, t] * np.exp(
            0.0001 + np.sqrt(vols[:, t]) * dW_s_adj[:, t]
        )

    return paths, vols


def generate_multi_asset_gbm(
    S0_vec=None,
    mu_vec=None,
    sigma_vec=None,
    correlation_matrix=None,
    n_steps=500,
    n_assets=5,
    seed=None,
):
    """Generate correlated GBM paths for multiple assets.

    Args:
        S0_vec: initial prices (n_assets,)
        mu_vec: drift coefficients (n_assets,)
        sigma_vec: volatility coefficients (n_assets,)
        correlation_matrix: (n_assets, n_assets) correlation matrix
        n_steps: number of steps
        n_assets: number of assets
        seed: random seed

    Returns:
        paths: (n_assets, n_steps) price arrays
    """
    if seed is not None:
        np.random.seed(seed)

    if S0_vec is None:
        S0_vec = np.full(n_assets, 100.0)
    if mu_vec is None:
        mu_vec = np.full(n_assets, 0.0001)
    if sigma_vec is None:
        sigma_vec = np.full(n_assets, 0.001)
    if correlation_matrix is None:
        correlation_matrix = np.eye(n_assets)

    # Cholesky decomposition of correlation matrix
    from scipy.linalg import cholesky
    L = cholesky(correlation_matrix, lower=True)

    paths = np.zeros((n_assets, n_steps))
    paths[:, 0] = S0_vec

    # Independent standard normals
    z = np.random.normal(0, 1, (n_assets, n_steps - 1))
    # Correlate them
    dW = L @ z

    for t in range(n_steps - 1):
        paths[:, t + 1] = paths[:, t] * np.exp(
            (mu_vec - 0.5 * sigma_vec ** 2).reshape(-1, 1) +
            (sigma_vec.reshape(-1, 1) * dW[:, t:t+1])
        )

    return paths
