import numpy as np
import pandas as pd

from tfmxai.robustness import _needed, _powerlaw_tau, bagging_backtest, percentile_coverage_reference


def _curve(a=0.5, b=0.5, ms=range(1, 6)):
    return pd.DataFrame({"M": list(ms), "tau": [1 - a * m ** (-b) for m in ms]})


def test_powerlaw_recovers_an_exact_curve():
    short = _curve(ms=range(1, 4))
    assert abs(_powerlaw_tau(short, 5) - (1 - 0.5 * 5 ** -0.5)) < 1e-9
    assert abs(_needed(short, 0.9) - 25) < 1e-6


def test_powerlaw_without_a_decline_never_reaches_the_target():
    flat = pd.DataFrame({"M": [1, 2, 3], "tau": [0.5, 0.5, 0.5]})
    assert np.isinf(_needed(flat)) or _needed(flat) > 1e6


def test_backtest_has_no_error_on_an_exact_curve():
    curves = _curve().assign(dataset="a", model="m", k=128)
    row = bagging_backtest(curves).iloc[0]
    assert abs(row.err_m5) < 1e-9
    assert abs(row.needed_short - row.needed_full) < 1e-6


def test_coverage_reference_for_unbiased_attributions():
    ref = percentile_coverage_reference(n_sim=1000, n_boot=300)
    assert 0.78 < ref["cover_bagged_ci"] < 0.90  # percentile bootstrap undercovers with M = 10
    assert ref["cover_single_range"] > 0.98
