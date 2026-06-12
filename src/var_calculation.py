"""
Value-at-Risk (VaR) Calculation using the Meixner Distribution
================================================================

Computes parametric VaR estimates from a fitted Meixner distribution and
provides a rolling-window VaR pipeline for backtesting.

Author: Vikrant Chandra
"""

import numpy as np
from .meixner_distribution import meixner, fit_meixner_mle


def meixner_var(alpha, beta, delta, m, confidence_level=0.99):
    """
    Compute the Value-at-Risk (VaR) implied by a fitted Meixner distribution.

    VaR is defined as the negative of the (1 - confidence_level) quantile of
    the return distribution, i.e., the loss that will not be exceeded with
    the given confidence level.

    Parameters
    ----------
    alpha, beta, delta, m : float
        Fitted Meixner distribution parameters.
    confidence_level : float, default 0.99
        VaR confidence level (e.g., 0.99 for 99% VaR).

    Returns
    -------
    float
        VaR value (positive number representing the loss magnitude).
    """
    from scipy.optimize import brentq
    from .meixner_distribution import meixner_pdf
    import numpy as np

    quantile_level = 1 - confidence_level

    # Theoretical mean/std to set a sensible numerical search bracket
    theo_mean = m + alpha * delta * np.tan(beta / 2)
    theo_std = np.sqrt((alpha ** 2) * delta / (np.cos(beta / 2) ** 2))

    lo = theo_mean - 20 * theo_std
    hi = theo_mean + 20 * theo_std

    def cdf_numeric(x):
        # Numerically integrate the PDF from lo to x
        from scipy.integrate import quad
        val, _ = quad(meixner_pdf, lo, x, args=(alpha, beta, delta, m), limit=200)
        return val

    def objective(x):
        return cdf_numeric(x) - quantile_level

    q = brentq(objective, lo, hi, xtol=1e-8)
    return -q  # VaR expressed as a positive loss magnitude


def rolling_meixner_var(returns, window=250, confidence_level=0.99, refit_every=20):
    """
    Compute rolling VaR estimates using a rolling-window Meixner MLE fit.

    For computational efficiency, the Meixner distribution is re-fitted
    every `refit_every` observations and the resulting VaR is held constant
    until the next refit (a common practical compromise in production risk
    systems).

    Parameters
    ----------
    returns : array_like
        1-D array of historical returns.
    window : int, default 250
        Rolling estimation window size (in observations / trading days).
    confidence_level : float, default 0.99
        VaR confidence level.
    refit_every : int, default 20
        Number of steps between successive MLE refits.

    Returns
    -------
    dict
        Dictionary with keys:
            'var_estimates' : ndarray of VaR estimates aligned to `returns`
                               (NaN for the initial window where no estimate
                               is available)
            'params_history': list of (index, params_tuple) for each refit
    """
    returns = np.asarray(returns, dtype=np.float64)
    n = len(returns)
    var_estimates = np.full(n, np.nan)
    params_history = []

    current_params = None

    for i in range(window, n):
        if (i - window) % refit_every == 0 or current_params is None:
            train_data = returns[i - window:i]
            fit_result = fit_meixner_mle(train_data)
            current_params = fit_result["params"]
            params_history.append((i, current_params))

        alpha, beta, delta, m = current_params
        var_estimates[i] = meixner_var(alpha, beta, delta, m, confidence_level)

    return {"var_estimates": var_estimates, "params_history": params_history}


def normal_var(returns, window=250, confidence_level=0.99):
    """
    Compute rolling VaR estimates using the standard parametric Normal
    (Gaussian) approach, for use as a benchmark comparison against the
    Meixner-based estimates.

    Parameters
    ----------
    returns : array_like
        1-D array of historical returns.
    window : int, default 250
        Rolling estimation window size.
    confidence_level : float, default 0.99
        VaR confidence level.

    Returns
    -------
    ndarray
        Array of VaR estimates aligned to `returns` (NaN for initial window).
    """
    from scipy.stats import norm

    returns = np.asarray(returns, dtype=np.float64)
    n = len(returns)
    var_estimates = np.full(n, np.nan)
    z = norm.ppf(1 - confidence_level)

    for i in range(window, n):
        train_data = returns[i - window:i]
        mu = np.mean(train_data)
        sigma = np.std(train_data, ddof=1)
        var_estimates[i] = -(mu + z * sigma)

    return var_estimates
