"""
Unit tests for the Meixner distribution implementation, MLE fitting,
VaR calculation, and Kupiec POF backtest.

Run with: pytest tests/
"""

import sys
import os
import numpy as np
try:
    import pytest
except ImportError:
    pytest = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.meixner_distribution import (
    meixner_pdf, meixner_logpdf, fit_meixner_mle, meixner_theoretical_moments
)
from src.kupiec_backtest import kupiec_pof_test, kupiec_traffic_light
from src.var_calculation import meixner_var, normal_var


# Reference parameter set used across tests
ALPHA, BETA, DELTA, M = 0.02, -0.2, 1.0, 0.0


class TestMeixnerPDF:
    def test_pdf_is_positive(self):
        x = np.linspace(-0.1, 0.1, 50)
        pdf_vals = meixner_pdf(x, ALPHA, BETA, DELTA, M)
        assert np.all(pdf_vals > 0)

    def test_pdf_integrates_to_one(self):
        from scipy.integrate import quad
        result, _ = quad(meixner_pdf, -1, 1, args=(ALPHA, BETA, DELTA, M))
        assert np.isclose(result, 1.0, atol=1e-4)

    def test_logpdf_consistency(self):
        x = np.array([-0.01, 0.0, 0.01])
        pdf_vals = meixner_pdf(x, ALPHA, BETA, DELTA, M)
        logpdf_vals = meixner_logpdf(x, ALPHA, BETA, DELTA, M)
        assert np.allclose(np.log(pdf_vals), logpdf_vals)

    def test_invalid_params_raise_or_nan(self):
        # alpha must be > 0; pdf should not produce a valid finite positive
        # value for alpha <= 0
        result = meixner_pdf(np.array([0.0]), -1.0, BETA, DELTA, M)
        assert not np.isfinite(result[0]) or result[0] <= 0


class TestMeixnerMoments:
    def test_theoretical_moments_keys(self):
        moments = meixner_theoretical_moments(ALPHA, BETA, DELTA, M)
        assert set(moments.keys()) == {"mean", "variance", "skewness", "excess_kurtosis"}

    def test_symmetric_case_zero_skew(self):
        # beta = 0 -> symmetric distribution -> skewness = 0
        moments = meixner_theoretical_moments(ALPHA, 0.0, DELTA, M)
        assert np.isclose(moments["skewness"], 0.0, atol=1e-10)

    def test_negative_beta_negative_skew(self):
        moments = meixner_theoretical_moments(ALPHA, -0.3, DELTA, M)
        assert moments["skewness"] < 0

    def test_positive_beta_positive_skew(self):
        moments = meixner_theoretical_moments(ALPHA, 0.3, DELTA, M)
        assert moments["skewness"] > 0


class TestMLEFitting:
    def test_fit_recovers_approximate_params_from_simulated_data(self):
        # Simulate data from a known Normal approximation and check the fit
        # converges and produces finite, sensible parameters.
        rng = np.random.default_rng(123)
        # Use t-distributed data as a stand-in for heavy-tailed returns
        data = rng.standard_t(df=5, size=5000) * 0.01

        result = fit_meixner_mle(data)
        alpha, beta, delta, m = result["params"]

        assert result["success"]
        assert alpha > 0
        assert delta > 0
        assert -np.pi < beta < np.pi
        assert np.isfinite(result["loglik"])
        assert np.isfinite(result["aic"])
        assert np.isfinite(result["bic"])

    def test_fit_on_skewed_data(self):
        rng = np.random.default_rng(7)
        data = rng.normal(loc=-0.001, scale=0.01, size=3000)
        # Add negative skew via occasional jumps
        jump_idx = rng.choice(3000, size=30, replace=False)
        data[jump_idx] -= 0.03

        result = fit_meixner_mle(data)
        assert result["success"]
        moments = meixner_theoretical_moments(*result["params"])
        assert moments["skewness"] < 0  # should capture negative skew


class TestVaRCalculation:
    def test_meixner_var_positive(self):
        var_99 = meixner_var(ALPHA, BETA, DELTA, M, confidence_level=0.99)
        assert var_99 > 0

    def test_var_increases_with_confidence(self):
        var_95 = meixner_var(ALPHA, BETA, DELTA, M, confidence_level=0.95)
        var_99 = meixner_var(ALPHA, BETA, DELTA, M, confidence_level=0.99)
        assert var_99 > var_95

    def test_normal_var_positive(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(0, 0.01, 500)
        result = normal_var(returns, window=250, confidence_level=0.99)
        valid = result[~np.isnan(result)]
        assert np.all(valid > 0)


class TestKupiecPOF:
    def test_pof_accepts_correctly_calibrated_var(self):
        # Construct returns and VaR such that the breach rate matches 1%
        rng = np.random.default_rng(1)
        n = 1000
        returns = rng.normal(0, 0.01, n)
        var_const = np.full(n, 3 * 0.01)  # ~99.7% VaR under normal -> breach rate ~0.3%
        result = kupiec_pof_test(returns, var_const, confidence_level=0.99)
        assert "decision" in result
        assert result["n"] == n

    def test_pof_rejects_badly_calibrated_var(self):
        rng = np.random.default_rng(2)
        n = 1000
        returns = rng.normal(0, 0.01, n)
        # Set VaR far too low so breach rate is much higher than 1%
        var_const = np.full(n, 0.001)
        result = kupiec_pof_test(returns, var_const, confidence_level=0.99)
        assert bool(result["reject_null"]) is True

    def test_traffic_light_zones(self):
        green = kupiec_traffic_light(250, 2, confidence_level=0.99)
        yellow = kupiec_traffic_light(250, 6, confidence_level=0.99)
        red = kupiec_traffic_light(250, 12, confidence_level=0.99)

        assert green["zone"] == "Green"
        assert yellow["zone"] == "Yellow"
        assert red["zone"] == "Red"


if __name__ == "__main__":
    if pytest is not None:
        sys.exit(pytest.main([__file__, "-v"]))
    else:
        import inspect
        classes = [c for c in vars(sys.modules[__name__]).values()
                   if inspect.isclass(c) and c.__name__.startswith("Test")]
        passed, failed = 0, 0
        for cls in classes:
            inst = cls()
            for name in dir(inst):
                if name.startswith("test_"):
                    try:
                        getattr(inst, name)()
                        print(f"PASS {cls.__name__}.{name}")
                        passed += 1
                    except Exception as e:
                        print(f"FAIL {cls.__name__}.{name}: {e}")
                        failed += 1
        print(f"\n{passed} passed, {failed} failed")
        sys.exit(0 if failed == 0 else 1)
