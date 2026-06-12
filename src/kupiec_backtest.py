"""
Kupiec Proportion of Failures (POF) Backtest
=============================================

Implements the Kupiec (1995) Proportion of Failures likelihood-ratio test,
used to validate the predictive accuracy of Value-at-Risk (VaR) models.

The test checks whether the observed number of VaR breaches (losses
exceeding the predicted VaR threshold) is statistically consistent with the
expected breach rate implied by the chosen confidence level.

Author: Vikrant Chandra
"""

import numpy as np
from scipy.stats import chi2


def kupiec_pof_test(returns, var_estimates, confidence_level=0.99):
    """
    Perform the Kupiec Proportion of Failures (POF) test.

    Parameters
    ----------
    returns : array_like
        Observed returns (e.g., daily portfolio P&L or returns).
    var_estimates : array_like
        VaR estimates corresponding to each return observation. VaR values
        should be expressed as negative numbers (loss thresholds) or
        positive thresholds consistently with `returns` sign convention.
        A "breach" / "exception" occurs when returns[i] < -var_estimates[i]
        (i.e., the realized loss exceeds the predicted VaR).
    confidence_level : float, default 0.99
        VaR confidence level (e.g., 0.99 for 99% VaR).

    Returns
    -------
    dict
        Dictionary with keys:
            'n'              : total number of observations
            'x'              : number of exceptions (breaches)
            'expected_rate'  : expected breach probability (1 - confidence_level)
            'observed_rate'  : observed breach rate (x / n)
            'lr_stat'        : likelihood-ratio test statistic
            'p_value'        : p-value from chi-squared(1) distribution
            'reject_null'    : True if model rejected at 95% significance (p < 0.05)
            'decision'       : human-readable verdict
    """
    returns = np.asarray(returns, dtype=np.float64)
    var_estimates = np.asarray(var_estimates, dtype=np.float64)

    n = len(returns)
    p = 1 - confidence_level

    # An exception occurs when the loss exceeds the VaR estimate
    # (returns more negative than -VaR)
    exceptions = returns < -np.abs(var_estimates)
    x = int(np.sum(exceptions))

    observed_rate = x / n

    # Likelihood ratio statistic (Kupiec, 1995)
    # LR_pof = -2 * ln[ (1-p)^(n-x) * p^x / ((1 - x/n)^(n-x) * (x/n)^x) ]
    if x == 0:
        # Avoid log(0): use limiting case
        log_l_null = n * np.log(1 - p)
        log_l_alt = 0.0  # (1 - 0/n)^n * (0/n)^0 -> 1^n * (0)^0 -> treated as 1
    elif x == n:
        log_l_null = n * np.log(p)
        log_l_alt = 0.0
    else:
        log_l_null = (n - x) * np.log(1 - p) + x * np.log(p)
        log_l_alt = (n - x) * np.log(1 - observed_rate) + x * np.log(observed_rate)

    lr_stat = -2 * (log_l_null - log_l_alt)
    lr_stat = max(lr_stat, 0.0)  # numerical guard

    p_value = 1 - chi2.cdf(lr_stat, df=1)
    reject_null = p_value < 0.05

    if reject_null:
        decision = "REJECT: Model VaR estimates are NOT statistically accurate (breach rate inconsistent with confidence level)."
    else:
        decision = "ACCEPT: Model VaR estimates are statistically consistent with the stated confidence level."

    return {
        "n": n,
        "x": x,
        "expected_rate": p,
        "observed_rate": observed_rate,
        "lr_stat": lr_stat,
        "p_value": p_value,
        "reject_null": reject_null,
        "decision": decision,
    }


def kupiec_traffic_light(n, x, confidence_level=0.99):
    """
    Basel Traffic Light approach for VaR backtesting (supplementary to Kupiec).

    Classifies the number of exceptions into Green / Yellow / Red zones
    based on the binomial distribution under the null hypothesis, using
    the standard Basel thresholds for a 250-day window.

    Parameters
    ----------
    n : int
        Number of observations (typically 250 trading days).
    x : int
        Number of observed exceptions.
    confidence_level : float, default 0.99
        VaR confidence level.

    Returns
    -------
    dict with keys 'zone' and 'cumulative_probability'
    """
    from scipy.stats import binom

    p = 1 - confidence_level
    cum_prob = 1 - binom.cdf(x - 1, n, p) if x > 0 else 1.0

    # Standard Basel zones for n=250, p=0.01
    if n == 250 and abs(confidence_level - 0.99) < 1e-9:
        if x <= 4:
            zone = "Green"
        elif x <= 9:
            zone = "Yellow"
        else:
            zone = "Red"
    else:
        # Generic classification based on cumulative probability
        if cum_prob > 0.05:
            zone = "Green"
        elif cum_prob > 0.0001:
            zone = "Yellow"
        else:
            zone = "Red"

    return {"zone": zone, "cumulative_probability": cum_prob, "exceptions": x}
