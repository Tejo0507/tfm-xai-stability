from itertools import permutations

import numpy as np

from tfmxai.shapley import all_masks, exact_shapley, explain, kernel_shapley, sample_kernel_masks


def _game(d, n, seed=0):
    """Random non-additive game: v(S) = a.S + b * (#S)^2 per point."""
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=(n, d)), rng.normal(size=n)
    return lambda masks: (masks.astype(float) @ a.T) + b[None] * masks.sum(1)[:, None] ** 2


def _brute_force(v, d):
    masks = all_masks(d)
    V = v(masks)
    lookup = {tuple(m): V[i] for i, m in enumerate(masks)}
    phi = np.zeros((V.shape[1], d))
    perms = list(permutations(range(d)))
    for order in perms:
        S = np.zeros(d, bool)
        for j in order:
            before = lookup[tuple(S)]
            S[j] = True
            phi[:, j] += lookup[tuple(S)] - before
    return phi / len(perms)


def test_exact_matches_permutation_definition():
    v = _game(4, 3)
    np.testing.assert_allclose(exact_shapley(v(all_masks(4)), 4), _brute_force(v, 4))


def test_exact_additive_game_recovers_coefficients():
    a = np.array([[1.0, -2.0, 0.5]])
    V = all_masks(3).astype(float) @ a.T
    np.testing.assert_allclose(exact_shapley(V, 3), a)


def test_kernel_efficiency_and_accuracy():
    d, v = 6, _game(6, 5, seed=1)
    exact = exact_shapley(v(all_masks(d)), d)
    masks = sample_kernel_masks(d, 4000, np.random.default_rng(0))
    phi = kernel_shapley(masks, v(masks))
    V = v(all_masks(d))
    np.testing.assert_allclose(phi.sum(1), V[-1] - V[0], atol=1e-8)
    np.testing.assert_allclose(phi, exact, atol=0.15)


def test_explain_switches_on_d():
    rng = np.random.default_rng(0)
    _, _, exact, n = explain(_game(4, 2), 4, exact_if_d_le=10, budget=64, rng=rng)
    assert exact and n == 16
    _, _, exact, n = explain(_game(12, 2), 12, exact_if_d_le=10, budget=64, rng=rng)
    assert not exact and n <= 64
