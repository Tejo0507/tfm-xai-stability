"""Refit value function: v(S) = logit p(true class | x_S) after fitting the model on the
context restricted to features S. The same definition is used for every model."""

import numpy as np

EPS = 1e-6


def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def prior_logits(y_ctx: np.ndarray, y_explain: np.ndarray, n_classes: int) -> np.ndarray:
    """v(empty): Laplace-smoothed class frequencies of the context."""
    freq = (np.bincount(y_ctx, minlength=n_classes) + 1) / (len(y_ctx) + n_classes)
    return to_logit(freq[y_explain])


def true_class_logits(proba: np.ndarray, present: np.ndarray, y: np.ndarray,
                      n_classes: int) -> np.ndarray:
    """Pick p(true class); classes absent from the context get probability EPS."""
    col = np.full(n_classes, -1)
    col[present] = np.arange(len(present))
    y_col = col[y]
    p = np.where(y_col >= 0, proba[np.arange(len(y)), np.maximum(y_col, 0)], EPS)
    return to_logit(p)


def varying_columns(X: np.ndarray) -> np.ndarray:
    """Columns that take more than one value in X (NaN counts as a value, all-NaN as constant)."""
    first = X[:1]
    same = (X == first) | (np.isnan(X) & np.isnan(first))
    return ~same.all(axis=0)


def coalition_values(model, X_ctx, y_ctx, X_exp, y_exp, masks, n_classes) -> np.ndarray:
    """Evaluate v(S) for every mask row; returns [n_masks, n_explain].

    A coalition whose features are all constant in the context carries no information,
    so it gets v(empty) for every model (TabPFN refuses to fit such a context)."""
    present, y_enc = np.unique(y_ctx, return_inverse=True)
    if len(present) < 2:
        raise ValueError("context contains a single class")
    varying = varying_columns(X_ctx)
    out = np.empty((len(masks), len(X_exp)))
    for r, S in enumerate(masks):
        if not (S & varying).any():
            out[r] = prior_logits(y_ctx, y_exp, n_classes)
            continue
        model.fit(X_ctx[:, S], y_enc)
        out[r] = true_class_logits(model.predict_proba(X_exp[:, S]), present, y_exp, n_classes)
    return out
