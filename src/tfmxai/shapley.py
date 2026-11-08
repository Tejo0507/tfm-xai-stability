"""Shapley values from a matrix of coalition values shared by all explained points.

Exact enumeration for small d; paired KernelSHAP (efficiency-constrained) otherwise.
Mask row r of `all_masks(d)` has feature j present iff bit j of r is set.
"""

from math import factorial

import numpy as np


def all_masks(d: int) -> np.ndarray:
    return ((np.arange(2**d)[:, None] >> np.arange(d)) & 1).astype(bool)


def exact_shapley(V: np.ndarray, d: int) -> np.ndarray:
    """V: [2**d, n] values in `all_masks(d)` order. Returns phi [n, d]."""
    ints = np.arange(2**d)
    sizes = all_masks(d).sum(1)
    w = np.array([factorial(s) * factorial(d - s - 1) / factorial(d) for s in range(d)])
    phi = np.empty((V.shape[1], d))
    for j in range(d):
        without = ints[((ints >> j) & 1) == 0]
        phi[:, j] = (w[sizes[without]][:, None] * (V[without | (1 << j)] - V[without])).sum(0)
    return phi


def sample_kernel_masks(d: int, budget: int, rng: np.random.Generator) -> np.ndarray:
    """Empty and full coalitions, then complement pairs with sizes from the Shapley kernel."""
    sizes = np.arange(1, d)
    p = (d - 1) / (sizes * (d - sizes))
    p /= p.sum()
    masks = [np.zeros(d, bool), np.ones(d, bool)]
    for s in rng.choice(sizes, size=(budget - 2) // 2, p=p):
        m = np.zeros(d, bool)
        m[rng.choice(d, size=s, replace=False)] = True
        masks += [m, ~m]
    return np.array(masks)


def kernel_shapley(masks: np.ndarray, V: np.ndarray) -> np.ndarray:
    """masks[0] empty, masks[1] full, rest kernel-sampled. Returns phi [n, d]."""
    v0, delta = V[0], V[1] - V[0]
    Z = masks[2:].astype(float)
    Y = V[2:] - v0
    # Eliminate the last feature using the efficiency constraint sum(phi) = delta.
    A = Z[:, :-1] - Z[:, [-1]]
    B = Y - Z[:, [-1]] * delta[None, :]
    phi_rest = np.linalg.lstsq(A, B, rcond=None)[0]
    phi = np.vstack([phi_rest, delta - phi_rest.sum(0)])
    return phi.T


def explain(value_fn, d: int, *, exact_if_d_le: int, budget: int,
            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, bool, int]:
    """value_fn(masks) -> [n_masks, n]. Returns (phi, v_full, exact, n_evaluated)."""
    if d <= exact_if_d_le:
        masks = all_masks(d)
        V = value_fn(masks)
        return exact_shapley(V, d), V[-1], True, len(masks)
    masks = sample_kernel_masks(d, budget, rng)
    uniq, inverse = np.unique(masks, axis=0, return_inverse=True)
    V = value_fn(uniq)[inverse.ravel()]
    return kernel_shapley(masks, V), V[1], False, len(uniq)
