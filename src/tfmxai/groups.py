"""Correlation-grouped Shapley values: features whose |Spearman rho| exceeds a threshold are
merged into one player, so credit is not split arbitrarily among near-duplicates."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def correlation_groups(X: np.ndarray, threshold: float = 0.7) -> np.ndarray:
    """Group label per feature (0..g-1, ordered by first feature). Average-linkage
    clustering on 1 - |Spearman rho|, cut at 1 - threshold. Computed on the whole pool so
    every context shares one partition."""
    d = X.shape[1]
    if d == 1:
        return np.zeros(1, int)
    corr = pd.DataFrame(X).corr(method="spearman").abs().fillna(0).to_numpy()
    dist = 1 - corr
    np.fill_diagonal(dist, 0)
    labels = fcluster(linkage(squareform(dist, checks=False), "average"),
                      t=1 - threshold, criterion="distance")
    _, first = np.unique(labels, return_index=True)
    order = {lab: i for i, lab in enumerate(labels[np.sort(first)])}
    return np.array([order[lab] for lab in labels])


def expand_masks(group_masks: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """[n_masks, g] group coalitions -> [n_masks, d] feature coalitions."""
    return group_masks[:, groups]
