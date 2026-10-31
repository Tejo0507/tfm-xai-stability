"""Dataset loading: OpenML classification tasks and synthetic data with an oracle."""

import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

MIN_CLASS_COUNT = 20  # classes rarer than this are dropped (e.g. nursery's "recommend")


@dataclass
class Dataset:
    name: str
    X_pool: np.ndarray
    y_pool: np.ndarray
    X_explain: np.ndarray
    y_explain: np.ndarray
    feature_names: list[str]
    n_classes: int
    # Synthetic only: oracle(X, y, masks) -> true-class logits [n_masks, n_points].
    oracle: Callable | None = field(default=None, repr=False)

    @property
    def d(self) -> int:
        return self.X_pool.shape[1]


def _encode(X: pd.DataFrame) -> np.ndarray:
    """Ordinal-encode categoricals; keep NaN as missing (all models handle or impute it)."""
    cols = []
    for c in X.columns:
        s = X[c]
        if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or s.dtype == bool:
            codes = pd.Categorical(s).codes.astype(float)
            codes[codes < 0] = np.nan
            cols.append(codes)
        else:
            cols.append(s.to_numpy(dtype=float))
    return np.column_stack(cols)


def _split(name, X, y, feature_names, n_explain, seed, oracle=None) -> Dataset:
    X_pool, X_exp, y_pool, y_exp = train_test_split(
        X, y, test_size=n_explain, stratify=y, random_state=seed
    )
    return Dataset(name, X_pool, y_pool, X_exp, y_exp, list(feature_names),
                   int(y.max()) + 1, oracle)


def load_openml(name: str, openml_id: int, n_explain: int = 200, seed: int = 0) -> Dataset:
    import openml

    if cache := os.environ.get("OPENML_CACHE_DIR"):
        openml.config.set_root_cache_directory(cache)
    ds = openml.datasets.get_dataset(
        openml_id, download_data=True, download_qualities=False,
        download_features_meta_data=False,
    )
    X, y, _, names = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")
    y = y.astype(str)
    counts = y.value_counts()
    keep = y.isin(counts[counts >= MIN_CLASS_COUNT].index).to_numpy()
    X, y = X[keep], y[keep]
    _, y_enc = np.unique(y.to_numpy(), return_inverse=True)
    return _split(name, _encode(X), y_enc, X.columns, n_explain, seed)


def make_synthetic(name: str, d: int, rho: float, n: int = 6000, n_explain: int = 200,
                   seed: int = 0) -> Dataset:
    """Equicorrelated Gaussian features, logistic labels with known beta.

    The oracle returns logit P(y = true class | x_S) under the true generative model,
    i.e. the value function every refit-based explainer is trying to estimate.
    """
    rng = np.random.default_rng(seed)
    cov = rho * np.ones((d, d)) + (1 - rho) * np.eye(d)
    beta = rng.choice([-1.0, 1.0], d) * np.linspace(1.5, 0.1, d)
    X = rng.multivariate_normal(np.zeros(d), cov, size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-X @ beta))).astype(int)
    oracle = _gaussian_logistic_oracle(cov, beta)
    return _split(name, X, y, [f"x{j}" for j in range(d)], n_explain, seed, oracle)


def _gaussian_logistic_oracle(cov: np.ndarray, beta: np.ndarray, n_nodes: int = 32):
    nodes, weights = np.polynomial.hermite_e.hermegauss(n_nodes)
    weights = weights / weights.sum()
    eps = 1e-6

    def oracle(X: np.ndarray, y: np.ndarray, masks: np.ndarray) -> np.ndarray:
        sign = np.where(y == 1, 1.0, -1.0)
        out = np.empty((len(masks), len(X)))
        for r, S in enumerate(masks):
            R = ~S
            if not S.any():
                mean, var = np.zeros(len(X)), beta @ cov @ beta
            else:
                # x_R | x_S ~ N(A x_S, C); z = x beta | x_S ~ N(mean, var)
                A = np.linalg.solve(cov[np.ix_(S, S)], cov[np.ix_(S, R)]).T
                C = cov[np.ix_(R, R)] - A @ cov[np.ix_(S, R)]
                mean = X[:, S] @ beta[S] + (X[:, S] @ A.T) @ beta[R]
                var = beta[R] @ C @ beta[R]
            z = mean[:, None] + np.sqrt(max(var, 0.0)) * nodes[None, :]
            p1 = (1 / (1 + np.exp(-z))) @ weights
            p_true = np.clip(np.where(sign > 0, p1, 1 - p1), eps, 1 - eps)
            out[r] = np.log(p_true) - np.log1p(-p_true)
        return out

    return oracle


def load_dataset(spec: dict, n_explain: int = 200, seed: int = 0) -> Dataset:
    if "openml_id" in spec:
        return load_openml(spec["name"], spec["openml_id"], n_explain, seed)
    return make_synthetic(spec["name"], spec["d"], spec["rho"], n_explain=n_explain, seed=seed)
