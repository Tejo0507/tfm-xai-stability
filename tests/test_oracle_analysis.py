import numpy as np

from tfmxai.oracle_analysis import bias_variance, coverage, oracle_tau


def test_unbiased_noise_is_all_variance_and_bagged_ci_covers():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(100, 5)) * np.arange(1, 6)
    phis = truth[None] + rng.normal(scale=0.5, size=(40, 100, 5))
    bv = bias_variance(phis, truth)
    assert bv["var_share"] > 0.95 and np.isclose(bv["var"], 0.25, rtol=0.1)
    cov = coverage(phis, truth, 500, rng)
    assert 0.8 < cov["cover_bagged_ci"] < 0.97 and cov["cover_single_range"] > 0.85
    t = oracle_tau(phis, truth)
    assert t["oracle_tau_bagged"] > t["oracle_tau_single"]


def test_pure_bias_is_never_covered():
    truth = np.ones((20, 3))
    phis = np.full((10, 20, 3), 2.0) + np.random.default_rng(1).normal(scale=0.01, size=(10, 20, 3))
    bv = bias_variance(phis, truth)
    assert bv["var_share"] < 0.01
    assert coverage(phis, truth, 200, np.random.default_rng(2))["cover_bagged_ci"] == 0.0
