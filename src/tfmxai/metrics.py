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


def variance_components(phis_seeds: np.ndarray) -> dict:
    """phis_seeds: [S seeds, M contexts, n, d]. Context variance of the seed-averaged phi
    vs. within-context variance across seeds (model randomness + explainer noise)."""
    ctx_var = phis_seeds.mean(0).var(0, ddof=1).mean()
    seed_var = phis_seeds.var(0, ddof=1).mean() if phis_seeds.shape[0] > 1 else np.nan
    return {"var_context": float(ctx_var), "var_seed": float(seed_var),
            "vr": float(ctx_var / seed_var) if seed_var > 0 else float("nan")}  # nan: deterministic model + exact SV


def oracle_error(phis: np.ndarray, phi_true: np.ndarray) -> dict:
    """Synthetic only: agreement of each context's attribution with the oracle."""
    rho = [np.mean([kendalltau(p[i], phi_true[i]).statistic for i in range(len(p))])
           for p in phis]
    return {"oracle_tau": float(np.nanmean(rho)),
            "oracle_mae": float(np.abs(phis - phi_true[None]).mean())}
