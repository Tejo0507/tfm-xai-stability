import numpy as np
import pytest

from tfmxai import metrics
from tfmxai.contexts import context_rng, sample_context
from tfmxai.data import make_synthetic
from tfmxai.models import make_model
from tfmxai.run import replicates_for
from tfmxai.shapley import all_masks, exact_shapley
from tfmxai.values import coalition_values, true_class_logits, varying_columns


def test_contexts_deterministic_and_valid():
    y = np.repeat([0, 1, 2], [500, 300, 20])
    a = sample_context(y, 128, "uniform", context_rng("ds", 128, 3))
    b = sample_context(y, 128, "uniform", context_rng("ds", 128, 3))
    c = sample_context(y, 128, "uniform", context_rng("ds", 128, 4))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert len(np.unique(a)) == 128 and a.max() < len(y)
    s = sample_context(y, 128, "stratified", context_rng("ds", 128, 0))
    assert set(y[s]) == {0, 1, 2}
    with pytest.raises(ValueError):
        sample_context(y, 10_000, "uniform", context_rng("ds", 1, 0))


def test_missing_class_gets_eps():
    proba = np.array([[0.7, 0.3], [0.2, 0.8]])
    out = true_class_logits(proba, np.array([0, 2]), np.array([1, 2]), n_classes=3)
    assert out[0] < -13 and out[1] == pytest.approx(np.log(0.8 / 0.2))


def test_refit_values_and_oracle_shapes():
    ds = make_synthetic("syn", d=4, rho=0.5, n=1200, n_explain=50)
    idx = sample_context(ds.y_pool, 256, "uniform", context_rng(ds.name, 256, 0))
    masks = all_masks(ds.d)
    V = coalition_values(make_model("logreg", 0), ds.X_pool[idx], ds.y_pool[idx],
                         ds.X_explain, ds.y_explain, masks, ds.n_classes)
    O = ds.oracle(ds.X_explain, ds.y_explain, masks)
    assert V.shape == O.shape == (16, 50) and np.isfinite(V).all()
    # A well-specified logistic model on 256 rows should track the oracle closely.
    phi, phi_true = exact_shapley(V, ds.d), exact_shapley(O, ds.d)
    assert np.corrcoef(phi.ravel(), phi_true.ravel())[0, 1] > 0.8


def test_constant_coalitions_get_the_prior():
    X = np.array([[1.0, np.nan, 0.0, np.nan], [1.0, np.nan, 1.0, 2.0], [1.0, np.nan, 0.0, 3.0]])
    assert varying_columns(X).tolist() == [False, False, True, True]
    X, y = X[:, [0, 2]], np.array([0, 1, 0])  # constant column, varying column
    masks = np.array([[0, 0], [1, 0], [1, 1]], bool)
    V = coalition_values(make_model("logreg", 0), X, y, X, y, masks, n_classes=2)
    assert np.allclose(V[1], V[0]) and not np.allclose(V[2], V[0])


def test_metrics_identical_contexts_are_perfectly_stable():
    phi = np.random.default_rng(0).normal(size=(10, 6))
    phis = np.stack([phi] * 4)
    assert metrics.local_kendall(phis) == pytest.approx(1.0)
    assert metrics.topk_jaccard(phis) == pytest.approx(1.0)
    assert metrics.sign_flip_rate(phis) == 0.0
    comps = metrics.variance_components(phis, np.stack([phis, phis + 0.01]))
    assert comps["var_context"] == pytest.approx(0.0, abs=1e-12)
    assert comps["var_seed"] > 0 and np.isnan(comps["var_explainer"])


def test_replicates_only_on_first_contexts():
    cfg = {"model_seeds": [0, 1], "explainer_seeds": [0, 1], "replicate_contexts": 3}
    assert replicates_for(cfg, 0, exact=True) == [(0, 0), (1, 0)]
    assert replicates_for(cfg, 0, exact=False) == [(0, 0), (1, 0), (0, 1)]
    assert replicates_for(cfg, 3, exact=False) == [(0, 0)]
