"""
Synthetic Equity Returns Generator
====================================

Generates a synthetic daily log-return series that mimics the statistical
properties of real equity returns: fat tails, negative skew, and volatility
clustering (via a simple GARCH-like process). This is used as sample input
data for the Meixner distribution fitting and backtesting pipeline when
live market data feeds are unavailable.

NOTE: For production use, replace this module with a real data loader
(e.g., reading historical adjusted-close prices from Bloomberg, Yahoo
Finance, or an internal market data warehouse) and computing log returns:
    returns = np.diff(np.log(prices))

Author: Vikrant Chandra
"""

import numpy as np
import pandas as pd


def generate_synthetic_returns(n_days=2000, seed=42, mu=0.0003, omega=1e-6,
                                arch=0.05, garch=0.90, skew_shock=-0.0015,
                                shock_prob=0.01):
    """
    Generate a synthetic daily return series using a GARCH(1,1)-style
    volatility process with occasional negative jump shocks to introduce
    realistic negative skewness and excess kurtosis (heavy tails).

    Parameters
    ----------
    n_days : int, default 2000
        Number of daily observations to generate (~8 years of trading days).
    seed : int, default 42
        Random seed for reproducibility.
    mu : float, default 0.0003
        Daily drift (mean return).
    omega, arch, garch : float
        GARCH(1,1) variance equation parameters:
            sigma2_t = omega + arch * eps_{t-1}^2 + garch * sigma2_{t-1}
    skew_shock : float, default -0.0015
        Mean magnitude of additional negative jump shocks.
    shock_prob : float, default 0.01
        Daily probability of a negative jump shock occurring.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ['date', 'price', 'return'] indexed by
        trading day.
    """
    rng = np.random.default_rng(seed)

    sigma2 = np.full(n_days, omega / (1 - arch - garch))
    eps = np.zeros(n_days)
    returns = np.zeros(n_days)

    for t in range(1, n_days):
        sigma2[t] = omega + arch * eps[t - 1] ** 2 + garch * sigma2[t - 1]
        z = rng.standard_t(df=5)  # t-distributed innovations for fat tails
        z = np.clip(z, -6, 6)
        eps[t] = np.sqrt(sigma2[t]) * z
        returns[t] = mu + eps[t]

        # Occasional negative jump (e.g. market shock / earnings surprise)
        if rng.random() < shock_prob:
            jump = skew_shock - abs(rng.normal(0, 0.01))
            returns[t] += jump

    # Build price series from returns (start at 100)
    prices = 100 * np.exp(np.cumsum(returns))
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    df = pd.DataFrame({
        "date": dates,
        "price": prices,
        "return": returns,
    })
    df.set_index("date", inplace=True)
    return df


if __name__ == "__main__":
    df = generate_synthetic_returns()
    out_path = "/home/claude/meixner-project/data/synthetic_equity_returns.csv"
    df.to_csv(out_path)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df.describe())
