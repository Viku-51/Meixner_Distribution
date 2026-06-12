"""
Main Analysis Pipeline
========================

Runs the full Meixner distribution implementation workflow:
  1. Load equity return data
  2. Fit Meixner distribution via MLE and compare to Normal fit
  3. Visualize the fitted PDFs against empirical histogram
  4. Compute Meixner-based and Normal-based rolling VaR
  5. Run Kupiec POF backtests on both VaR series
  6. Save all results and plots to the results/ directory

Author: Vikrant Chandra
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, skew, kurtosis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.meixner_distribution import (
    meixner, fit_meixner_mle, meixner_theoretical_moments
)
from src.var_calculation import rolling_meixner_var, normal_var, meixner_var
from src.kupiec_backtest import kupiec_pof_test, kupiec_traffic_light

sns.set_theme(style="whitegrid")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic_equity_returns.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    # -------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    returns = df["return"].dropna().values
    print(f"Loaded {len(returns)} daily return observations.")
    print(f"  Mean: {np.mean(returns):.6f}")
    print(f"  Std Dev: {np.std(returns):.6f}")
    print(f"  Skewness: {skew(returns):.4f}")
    print(f"  Excess Kurtosis: {kurtosis(returns):.4f}\n")

    # -------------------------------------------------------------
    # 2. Fit Meixner distribution via MLE (full sample)
    # -------------------------------------------------------------
    fit_result = fit_meixner_mle(returns, verbose=False)
    alpha, beta, delta, m = fit_result["params"]
    print("Meixner MLE fit (full sample):")
    print(f"  alpha = {alpha:.6f}")
    print(f"  beta  = {beta:.6f}")
    print(f"  delta = {delta:.6f}")
    print(f"  m     = {m:.6f}")
    print(f"  Log-Likelihood = {fit_result['loglik']:.2f}")
    print(f"  AIC = {fit_result['aic']:.2f}, BIC = {fit_result['bic']:.2f}")
    print(f"  Converged: {fit_result['success']} ({fit_result['message']})\n")

    theo_moments = meixner_theoretical_moments(alpha, beta, delta, m)
    print("Theoretical moments implied by fitted Meixner params:")
    for k, v in theo_moments.items():
        print(f"  {k}: {v:.6f}")
    print()

    # -------------------------------------------------------------
    # 3. Compare Meixner fit vs Normal fit vs Empirical histogram
    # -------------------------------------------------------------
    mu_norm, sigma_norm = np.mean(returns), np.std(returns, ddof=1)

    x_grid = np.linspace(returns.min() * 1.1, returns.max() * 1.1, 1000)
    meixner_pdf_vals = meixner.pdf(x_grid, alpha, beta, delta, m)
    normal_pdf_vals = norm.pdf(x_grid, mu_norm, sigma_norm)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(returns, bins=80, density=True, alpha=0.4, color="steelblue",
            label="Empirical Returns (Histogram)")
    ax.plot(x_grid, meixner_pdf_vals, color="crimson", lw=2,
            label="Meixner Fit (MLE)")
    ax.plot(x_grid, normal_pdf_vals, color="gray", lw=2, linestyle="--",
            label="Normal Fit")
    ax.set_title("Equity Returns: Meixner vs Normal Distribution Fit")
    ax.set_xlabel("Daily Log Return")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "meixner_vs_normal_pdf.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved distribution comparison plot -> {fig_path}")

    # Log-scale tail comparison (key for showing heavy-tail capture)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(returns, bins=80, density=True, alpha=0.4, color="steelblue",
            label="Empirical Returns (Histogram)")
    ax.plot(x_grid, meixner_pdf_vals, color="crimson", lw=2, label="Meixner Fit")
    ax.plot(x_grid, normal_pdf_vals, color="gray", lw=2, linestyle="--", label="Normal Fit")
    ax.set_yscale("log")
    ax.set_title("Tail Comparison (Log Scale): Meixner vs Normal")
    ax.set_xlabel("Daily Log Return")
    ax.set_ylabel("Density (log scale)")
    ax.legend()
    fig.tight_layout()
    fig_path_log = os.path.join(RESULTS_DIR, "meixner_vs_normal_tails_log.png")
    fig.savefig(fig_path_log, dpi=150)
    plt.close(fig)
    print(f"Saved log-scale tail comparison plot -> {fig_path_log}")

    # -------------------------------------------------------------
    # 4. Rolling VaR: Meixner vs Normal
    # -------------------------------------------------------------
    confidence_level = 0.99
    window = 250

    print(f"\nComputing rolling {int(confidence_level*100)}% VaR "
          f"(window={window} days)...")
    meixner_result = rolling_meixner_var(returns, window=window,
                                          confidence_level=confidence_level,
                                          refit_every=20)
    meixner_var_series = meixner_result["var_estimates"]
    normal_var_series = normal_var(returns, window=window,
                                    confidence_level=confidence_level)

    # -------------------------------------------------------------
    # 5. Kupiec POF backtests
    # -------------------------------------------------------------
    valid = ~np.isnan(meixner_var_series)
    test_returns = returns[valid]
    meixner_var_valid = meixner_var_series[valid]
    normal_var_valid = normal_var_series[valid]

    meixner_kupiec = kupiec_pof_test(test_returns, meixner_var_valid, confidence_level)
    normal_kupiec = kupiec_pof_test(test_returns, normal_var_valid, confidence_level)

    print("\n--- Kupiec POF Backtest Results (Meixner VaR) ---")
    for k, v in meixner_kupiec.items():
        print(f"  {k}: {v}")

    print("\n--- Kupiec POF Backtest Results (Normal VaR) ---")
    for k, v in normal_kupiec.items():
        print(f"  {k}: {v}")

    # Basel traffic light (using 250-day sub-window of most recent data)
    n_recent = min(250, len(test_returns))
    recent_returns = test_returns[-n_recent:]
    recent_meixner_var = meixner_var_valid[-n_recent:]
    recent_normal_var = normal_var_valid[-n_recent:]

    meixner_exceptions = int(np.sum(recent_returns < -np.abs(recent_meixner_var)))
    normal_exceptions = int(np.sum(recent_returns < -np.abs(recent_normal_var)))

    meixner_tl = kupiec_traffic_light(n_recent, meixner_exceptions, confidence_level)
    normal_tl = kupiec_traffic_light(n_recent, normal_exceptions, confidence_level)

    print(f"\nBasel Traffic Light (last {n_recent} days):")
    print(f"  Meixner VaR: {meixner_tl}")
    print(f"  Normal VaR:  {normal_tl}")

    # -------------------------------------------------------------
    # 6. Plot VaR vs Returns with exceptions highlighted
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    dates = df.index[valid]

    axes[0].plot(dates, test_returns, color="black", lw=0.7, label="Daily Returns")
    axes[0].plot(dates, -meixner_var_valid, color="crimson", lw=1.3,
                  label=f"Meixner {int(confidence_level*100)}% VaR (threshold)")
    breach_mask_m = test_returns < -np.abs(meixner_var_valid)
    axes[0].scatter(dates[breach_mask_m], test_returns[breach_mask_m],
                     color="red", marker="x", s=40, zorder=5, label="VaR Breach")
    axes[0].set_title("Meixner-based VaR Backtest")
    axes[0].set_ylabel("Daily Return")
    axes[0].legend(loc="lower left", fontsize=9)

    axes[1].plot(dates, test_returns, color="black", lw=0.7, label="Daily Returns")
    axes[1].plot(dates, -normal_var_valid, color="gray", lw=1.3, linestyle="--",
                  label=f"Normal {int(confidence_level*100)}% VaR (threshold)")
    breach_mask_n = test_returns < -np.abs(normal_var_valid)
    axes[1].scatter(dates[breach_mask_n], test_returns[breach_mask_n],
                     color="red", marker="x", s=40, zorder=5, label="VaR Breach")
    axes[1].set_title("Normal-based VaR Backtest")
    axes[1].set_ylabel("Daily Return")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    fig_path_var = os.path.join(RESULTS_DIR, "var_backtest_comparison.png")
    fig.savefig(fig_path_var, dpi=150)
    plt.close(fig)
    print(f"\nSaved VaR backtest comparison plot -> {fig_path_var}")

    # -------------------------------------------------------------
    # 7. Save summary results to text file
    # -------------------------------------------------------------
    summary_path = os.path.join(RESULTS_DIR, "summary_results.txt")
    with open(summary_path, "w") as f:
        f.write("Meixner Distribution Implementation - Summary Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Sample size: {len(returns)} daily returns\n")
        f.write(f"Empirical mean: {np.mean(returns):.6f}\n")
        f.write(f"Empirical std dev: {np.std(returns):.6f}\n")
        f.write(f"Empirical skewness: {skew(returns):.4f}\n")
        f.write(f"Empirical excess kurtosis: {kurtosis(returns):.4f}\n\n")

        f.write("Meixner MLE Fitted Parameters:\n")
        f.write(f"  alpha = {alpha:.6f}\n")
        f.write(f"  beta  = {beta:.6f}\n")
        f.write(f"  delta = {delta:.6f}\n")
        f.write(f"  m     = {m:.6f}\n")
        f.write(f"  Log-Likelihood = {fit_result['loglik']:.2f}\n")
        f.write(f"  AIC = {fit_result['aic']:.2f}\n")
        f.write(f"  BIC = {fit_result['bic']:.2f}\n\n")

        f.write("Kupiec POF Test - Meixner VaR:\n")
        for k, v in meixner_kupiec.items():
            f.write(f"  {k}: {v}\n")

        f.write("\nKupiec POF Test - Normal VaR:\n")
        for k, v in normal_kupiec.items():
            f.write(f"  {k}: {v}\n")

        f.write(f"\nBasel Traffic Light (last {n_recent} days):\n")
        f.write(f"  Meixner VaR: {meixner_tl}\n")
        f.write(f"  Normal VaR:  {normal_tl}\n")

    print(f"\nSaved summary results -> {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
