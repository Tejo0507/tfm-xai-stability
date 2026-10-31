"""Context samplers. Contexts depend only on (dataset, k, context id), never on the model,
so every model and baseline is explained on identical row sets."""

import zlib

import numpy as np
from sklearn.model_selection import train_test_split


def context_rng(dataset: str, k: int, context_id: int) -> np.random.Generator:
    return np.random.default_rng([zlib.crc32(dataset.encode()), k, context_id])


def sample_context(y_pool: np.ndarray, k: int, strategy: str,
                   rng: np.random.Generator) -> np.ndarray:
    n = len(y_pool)
    if k > n:
        raise ValueError(f"context size {k} exceeds pool size {n}")
    if strategy == "uniform":
        idx = rng.choice(n, size=k, replace=False)
    elif strategy == "stratified":
        if k == n:
            idx = np.arange(n)
        else:
            idx, _ = train_test_split(np.arange(n), train_size=k, stratify=y_pool,
                                      random_state=int(rng.integers(2**31)))
    else:
        raise ValueError(f"unknown sampling strategy: {strategy}")
    return np.sort(idx)
