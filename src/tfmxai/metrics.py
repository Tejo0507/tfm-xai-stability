"""Stability metrics over attributions shaped [M contexts, n points, d features]."""

from itertools import combinations

import numpy as np
from scipy.stats import kendalltau


def local_kendall(phis: np.ndarray) -> float:
    """Primary metric: mean pairwise Kendall tau_b of per-point |phi| rankings."""
    M, n, _ = phis.shape
    absphi = np.abs(phis)
    taus = [kendalltau(absphi[a, i], absphi[b, i]).statistic
            for a, b in combinations(range(M), 2) for i in range(n)]
    return float(np.nanmean(taus))


def global_kendall(phis: np.ndarray) -> float:
    """Pairwise Kendall tau of mean-|phi| global importance across contexts."""
    imp = np.abs(phis).mean(1)
    taus = [kendalltau(imp[a], imp[b]).statistic for a, b in combinations(range(len(imp)), 2)]
    return float(np.nanmean(taus))


def topk_jaccard(phis: np.ndarray, k: int = 3) -> float:
    M, n, d = phis.shape
    k = min(k, d)
    top = np.argsort(-np.abs(phis), axis=2)[:, :, :k]
    vals = []
    for a, b in combinations(range(M), 2):
        for i in range(n):
            sa, sb = set(top[a, i]), set(top[b, i])
            vals.append(len(sa & sb) / len(sa | sb))
    return float(np.mean(vals))


def sign_flip_rate(phis: np.ndarray, min_share: float = 0.05) -> float:
    """Among (point, feature) pairs carrying >= min_share of the point's mean total |phi|,
    the fraction whose sign is not the same in every context."""
    absphi = np.abs(phis)
    share = absphi.mean(0) / np.maximum(absphi.mean(0).sum(1, keepdims=True), 1e-12)
    relevant = share >= min_share
    signs = np.sign(phis)
    flipped = (signs.max(0) > 0) & (signs.min(0) < 0)
    return float(flipped[relevant].mean()) if relevant.any() else float("nan")


def variance_components(phis: np.ndarray, model_reps: np.ndarray | None = None,
                        explainer_reps: np.ndarray | None = None) -> dict:
    """phis: [M contexts, n, d], one run per context. *_reps: [R replicates, C contexts, n, d],
    the same contexts rerun with another model seed / explainer seed. Variances are
    per-(point, feature), averaged. vr = context variance / model-seed variance."""
    within = lambda reps: float(reps.var(0, ddof=1).mean()) if reps is not None else np.nan  # noqa: E731
    ctx_var, seed_var = float(phis.var(0, ddof=1).mean()), within(model_reps)
    return {"var_context": ctx_var, "var_seed": seed_var, "var_explainer": within(explainer_reps),
            "vr": ctx_var / seed_var if seed_var > 0 else np.nan}  # nan: deterministic model


def oracle_error(phis: np.ndarray, phi_true: np.ndarray) -> dict:
    """Synthetic only: agreement of each context's attribution with the oracle."""
    rho = [np.mean([kendalltau(p[i], phi_true[i]).statistic for i in range(len(p))])
           for p in phis]
    return {"oracle_tau": float(np.nanmean(rho)),
            "oracle_mae": float(np.abs(phis - phi_true[None]).mean())}
