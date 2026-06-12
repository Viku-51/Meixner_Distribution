"""
Meixner Distribution Implementation for Equity Risk Modeling
================================================================

This module implements the probability density function (PDF), cumulative
distribution function (CDF), and Maximum Likelihood Estimation (MLE) fitting
routine for the Meixner distribution, used to model heavy-tailed financial
asset returns.

The Meixner distribution is a member of the family of generalized hyperbolic
distributions. It is parameterized by four parameters: alpha (scale), beta
(skewness/asymmetry), delta (shape), and m (location). It is well suited to
capturing the leptokurtosis (fat tails) and skewness observed in daily
financial returns, which the Gaussian distribution fails to model.

PDF:
    f(x; alpha, beta, delta, m) =
        ((2 * cos(beta/2))^(2*delta)) / (2 * alpha * pi * Gamma(2*delta))
        * exp(beta * (x - m) / alpha)
        * |Gamma(delta + i*(x - m)/alpha)|^2

Author: Vikrant Chandra
"""

import numpy as np
from scipy.special import gamma, gammaln
from scipy.optimize import minimize
from scipy.stats import rv_continuous


# ---------------------------------------------------------------------------
# Core Meixner PDF / log-PDF
# ---------------------------------------------------------------------------

def meixner_logpdf(x, alpha, beta, delta, m):
    """
    Compute the log-density of the Meixner distribution at point(s) x.

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the log-density.
    alpha : float
        Scale parameter (alpha > 0).
    beta : float
        Skewness/asymmetry parameter (-pi < beta < pi).
    delta : float
        Shape parameter (delta > 0).
    m : float
        Location parameter.

    Returns
    -------
    ndarray
        Log-density values evaluated at x.
    """
    x = np.asarray(x, dtype=np.float64)
    z = (x - m) / alpha

    # log of the normalizing constant:
    #   (2*cos(beta/2))^(2*delta) / (2*alpha*pi*Gamma(2*delta))
    log_norm = (2 * delta * np.log(2 * np.cos(beta / 2))
                - np.log(2 * alpha * np.pi)
                - gammaln(2 * delta))

    # log |Gamma(delta + i*z)|^2 = 2 * Re[log Gamma(delta + i*z)]
    # Using scipy's complex-argument gammaln (loggamma) via gammaln on complex input
    from scipy.special import loggamma
    log_gamma_term = 2 * np.real(loggamma(delta + 1j * z))

    return log_norm + beta * z + log_gamma_term


def meixner_pdf(x, alpha, beta, delta, m):
    """
    Compute the Meixner probability density function at point(s) x.

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the density.
    alpha, beta, delta, m : float
        Meixner distribution parameters (see meixner_logpdf for definitions).

    Returns
    -------
    ndarray
        Density values evaluated at x.
    """
    return np.exp(meixner_logpdf(x, alpha, beta, delta, m))


# ---------------------------------------------------------------------------
# scipy rv_continuous wrapper (enables .cdf, .ppf, .rvs, etc.)
# ---------------------------------------------------------------------------

class meixner_gen(rv_continuous):
    """
    Continuous random variable class for the Meixner distribution,
    built on scipy.stats.rv_continuous.

    Usage
    -----
    >>> meixner = meixner_gen(name="meixner")
    >>> meixner.pdf(0.0, alpha=0.02, beta=-0.3, delta=1.2, m=0.0005)
    """

    def _pdf(self, x, alpha, beta, delta, m):
        return meixner_pdf(x, alpha, beta, delta, m)

    def _argcheck(self, alpha, beta, delta, m):
        return (alpha > 0) and (delta > 0) and (-np.pi < beta < np.pi)


meixner = meixner_gen(name="meixner", a=-np.inf, b=np.inf)


# ---------------------------------------------------------------------------
# Moment-based starting values
# ---------------------------------------------------------------------------

def meixner_moments_to_params(mean, var, skew, exkurt):
    """
    Convert sample moments (mean, variance, skewness, excess kurtosis) into
    approximate starting parameters (alpha, beta, delta, m) for the Meixner
    distribution, using the analytic moment relationships.

    These provide a good initial guess for the MLE optimizer.

    Parameters
    ----------
    mean, var, skew, exkurt : float
        Sample moments of the data.

    Returns
    -------
    tuple of float
        (alpha, beta, delta, m) starting parameter guesses.
    """
    # exkurt = 3 + (2 - cos(beta)) / delta  =>  delta = (2 - cos(beta)) / (exkurt - 3)
    # skew = sin(beta/2) * sqrt(2 / delta)  (sign/scale relationship)
    # Use a robust numerical search over beta to match skew & kurtosis jointly.
    from scipy.optimize import brentq

    exkurt = max(exkurt, 1e-3)  # guard against non-positive excess kurtosis

    def kurt_residual(beta):
        delta = (2 - np.cos(beta)) / exkurt
        skew_pred = np.sin(beta / 2) * np.sqrt(2.0 / delta)
        return skew_pred - skew

    # search beta in (-pi, pi), avoiding singularities near +-pi
    try:
        beta = brentq(kurt_residual, -np.pi + 1e-3, np.pi - 1e-3)
    except ValueError:
        beta = -0.1 if skew < 0 else 0.1

    delta = (2 - np.cos(beta)) / exkurt
    delta = max(delta, 1e-3)

    # variance = alpha^2 * delta / cos^2(beta/2)
    alpha = np.sqrt(var * np.cos(beta / 2) ** 2 / delta)
    alpha = max(alpha, 1e-6)

    # mean = m + alpha * delta * tan(beta/2)
    m = mean - alpha * delta * np.tan(beta / 2)

    return alpha, beta, delta, m


# ---------------------------------------------------------------------------
# Maximum Likelihood Estimation
# ---------------------------------------------------------------------------

def fit_meixner_mle(data, initial_guess=None, verbose=False):
    """
    Fit Meixner distribution parameters to data using Maximum Likelihood
    Estimation (MLE).

    Parameters
    ----------
    data : array_like
        1-D array of observed returns (e.g., daily log returns).
    initial_guess : tuple of float, optional
        Starting values (alpha, beta, delta, m). If None, computed
        automatically from sample moments.
    verbose : bool, default False
        If True, print optimizer progress information.

    Returns
    -------
    dict
        Dictionary with keys:
            'params'   : (alpha, beta, delta, m) fitted parameters
            'loglik'   : final log-likelihood value
            'aic'      : Akaike Information Criterion
            'bic'      : Bayesian Information Criterion
            'success'  : optimizer convergence flag
            'message'  : optimizer message
    """
    data = np.asarray(data, dtype=np.float64)
    data = data[np.isfinite(data)]
    n = len(data)

    if initial_guess is None:
        from scipy.stats import skew as sk, kurtosis as kt
        mean = np.mean(data)
        var = np.var(data)
        skew = sk(data)
        exkurt = kt(data)  # excess kurtosis (Fisher)
        initial_guess = meixner_moments_to_params(mean, var, skew, exkurt + 3)

    def neg_loglik(params):
        alpha, beta, delta, m = params
        if alpha <= 0 or delta <= 0 or not (-np.pi < beta < np.pi):
            return 1e10
        ll = meixner_logpdf(data, alpha, beta, delta, m)
        if not np.all(np.isfinite(ll)):
            return 1e10
        return -np.sum(ll)

    bounds = [(1e-8, None), (-np.pi + 1e-6, np.pi - 1e-6), (1e-6, None), (None, None)]

    result = minimize(
        neg_loglik,
        x0=initial_guess,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "disp": verbose},
    )

    alpha, beta, delta, m = result.x
    loglik = -result.fun
    k = 4  # number of parameters
    aic = 2 * k - 2 * loglik
    bic = k * np.log(n) - 2 * loglik

    return {
        "params": (alpha, beta, delta, m),
        "loglik": loglik,
        "aic": aic,
        "bic": bic,
        "success": result.success,
        "message": result.message,
    }


# ---------------------------------------------------------------------------
# Theoretical moments (for diagnostics)
# ---------------------------------------------------------------------------

def meixner_theoretical_moments(alpha, beta, delta, m):
    """
    Compute the theoretical mean, variance, skewness, and excess kurtosis
    of the Meixner distribution given its parameters.

    Returns
    -------
    dict with keys: 'mean', 'variance', 'skewness', 'excess_kurtosis'
    """
    mean = m + alpha * delta * np.tan(beta / 2)
    variance = (alpha ** 2) * delta / (np.cos(beta / 2) ** 2)
    skewness = np.sin(beta / 2) * np.sqrt(2.0 / delta)
    excess_kurtosis = 3 + (2 - np.cos(beta)) / delta - 3 + 3  # = (2-cos(beta))/delta + (term)
    # Correct formula: excess kurtosis = (2 - cos(beta)) / delta
    excess_kurtosis = (2 - np.cos(beta)) / delta

    return {
        "mean": mean,
        "variance": variance,
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
    }
