import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from tfmxai.analysis import bagging_curve, contexts_needed, paired_tests, vr_bootstrap
from tfmxai.metrics import kendall_rows


def test_kendall_rows_matches_scipy_with_ties():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(30, 6)), rng.normal(size=(30, 6))
    a[:, 1] = a[:, 0]  # ties
    ours = kendall_rows(a, b)
    ref = [kendalltau(a[i], b[i]).statistic for i in range(30)]
    assert np.allclose(ours, ref)


def test_vr_bootstrap_brackets_point_estimate():
    rng = np.random.default_rng(1)
    signal = rng.normal(size=(10, 50, 4))                 # context variance 1
    reps = rng.normal(scale=0.1, size=(2, 3, 50, 4))      # seed variance 0.01
    vr, lo, hi = vr_bootstrap(signal, reps, 500, rng)
    assert lo <= vr <= hi and 30 < vr < 300


def test_bagging_improves_and_identical_contexts_are_perfect():
    rng = np.random.default_rng(2)
    truth = rng.normal(size=(40, 5)) * np.arange(1, 6)
    noisy = truth[None] + rng.normal(scale=2.0, size=(10, 40, 5))
    curve = bagging_curve(noisy, 20, rng)
    assert curve.tau.iloc[-1] > curve.tau.iloc[0]
    same = bagging_curve(np.repeat(truth[None], 10, 0), 5, rng)
    assert np.allclose(same.tau, 1.0)
    assert contexts_needed(same) == (1.0, False)
    M, extrap = contexts_needed(pd.DataFrame({"M": [1, 2, 3, 4, 5],
                                              "tau": [0.5, 0.6, 0.66, 0.7, 0.73]}))
    assert extrap and M > 5


def test_paired_tests_detects_a_consistent_gap():
    rows = [{"dataset": f"d{i}", "k": k, "model": m, "tau_local": base + gap}
            for i in range(12) for k in (128, 512)
            for m, gap in (("tabpfn_v2_5", 0.1), ("xgboost", 0.0))
            for base in [0.3 + 0.03 * i]]
    out = paired_tests(pd.DataFrame(rows), "tau_local", 200, np.random.default_rng(0))
    assert len(out) == 2 and (out.wins == 12).all() and (out.p_holm < 0.01).all()
