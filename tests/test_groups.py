import numpy as np

from tfmxai.contexts import context_rng, sample_context
from tfmxai.data import make_synthetic
from tfmxai.groups import correlation_groups, expand_masks
from tfmxai.models import make_model
from tfmxai.shapley import all_masks, exact_shapley
from tfmxai.values import coalition_values


def test_near_duplicates_are_merged_and_labels_ordered():
    rng = np.random.default_rng(0)
    a, b, c = rng.normal(size=(3, 500))
    X = np.column_stack([a, b, a + 0.05 * rng.normal(size=500), c, b + 0.05 * rng.normal(size=500)])
    assert correlation_groups(X, 0.7).tolist() == [0, 1, 0, 2, 1]
    assert correlation_groups(rng.normal(size=(500, 4)), 0.7).tolist() == [0, 1, 2, 3]


def test_grouped_shapley_is_efficient_and_identity_groups_change_nothing():
    ds = make_synthetic("syn", d=4, rho=0.9, n=1200, n_explain=30)
    idx = sample_context(ds.y_pool, 256, "uniform", context_rng(ds.name, 256, 0))
    value = lambda m: coalition_values(make_model("logreg", 0), ds.X_pool[idx],  # noqa: E731
                                       ds.y_pool[idx], ds.X_explain, ds.y_explain, m, 2)
    groups = np.array([0, 0, 1, 2])
    V = value(expand_masks(all_masks(3), groups))
    phi = exact_shapley(V, 3)
    assert phi.shape == (30, 3) and np.allclose(phi.sum(1), V[-1] - V[0])
    ident = np.arange(4)
    assert np.allclose(exact_shapley(value(expand_masks(all_masks(4), ident)), 4),
                       exact_shapley(value(all_masks(4)), 4))
