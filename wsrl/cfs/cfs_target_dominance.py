"""REDQ target-dominance diagnostics for CFS-D."""

from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np


def estimate_min_win_rate(
    q_values: np.ndarray,
    *,
    critic_subsample_size: int = 2,
    num_samples: int = 20_000,
    seed: int = 0,
    max_exact_tuples: int = 10_000,
) -> np.ndarray:
    """Estimate how often each head wins REDQ's min-over-subset target.

    Args:
        q_values: Array of shape ``(num_heads, num_examples)`` containing
            per-head Q-values at target states/actions.
        critic_subsample_size: REDQ target subset size, usually 2.
        num_samples: Number of random subsets for the Monte-Carlo fallback.
            For the common H=10, K=2 case, the function computes the exact
            with-replacement expectation over all H**K tuples instead.
        seed: RNG seed for the Monte-Carlo fallback.
        max_exact_tuples: Use exact enumeration if ``H ** K`` is at most this.

    Returns:
        Array ``d_h`` of shape ``(num_heads,)``. Values sum to 1.
    """

    q_values = np.asarray(q_values, dtype=np.float64)
    if q_values.ndim != 2:
        raise ValueError(f"q_values must have shape (H, N); got {q_values.shape}.")
    num_heads, num_examples = q_values.shape
    if num_heads <= 0 or num_examples <= 0:
        raise ValueError(f"q_values must be non-empty; got {q_values.shape}.")

    k = int(critic_subsample_size)
    if k <= 0:
        raise ValueError(f"critic_subsample_size must be positive, got {k}.")

    counts = np.zeros(num_heads, dtype=np.float64)
    exact_tuples = num_heads**k

    if exact_tuples <= int(max_exact_tuples):
        total_cases = 0
        for subset in itertools.product(range(num_heads), repeat=k):
            subset = np.asarray(subset, dtype=np.int64)
            subset_q = q_values[subset]  # (k, N)
            winner_pos = np.argmin(subset_q, axis=0)  # (N,)
            winners = subset[winner_pos]
            counts += np.bincount(winners, minlength=num_heads)
            total_cases += num_examples
        return counts / max(float(total_cases), 1.0)

    # Monte-Carlo fallback for unusually large ensembles/subsets.
    rng = np.random.default_rng(seed)
    remaining = int(num_samples)
    total_cases = 0
    chunk = 512
    while remaining > 0:
        m = min(chunk, remaining)
        subsets = rng.integers(0, num_heads, size=(m, k), endpoint=False)
        for subset in subsets:
            subset_q = q_values[subset]
            winner_pos = np.argmin(subset_q, axis=0)
            winners = subset[winner_pos]
            counts += np.bincount(winners, minlength=num_heads)
            total_cases += num_examples
        remaining -= m
    return counts / max(float(total_cases), 1.0)


def compute_eta(
    rho: Sequence[float], dominance: Sequence[float], eps: float = 1e-8
) -> np.ndarray:
    """Compute target-dominant conservatism influence.

    ``eta_h = rho_h * (d_h / (1/H))``.
    A head with above-uniform target-min dominance therefore receives a larger
    influence score than a head with the same footprint but low target control.
    """

    rho = np.asarray(rho, dtype=np.float64).reshape(-1)
    dominance = np.asarray(dominance, dtype=np.float64).reshape(-1)
    if rho.size != dominance.size:
        raise ValueError(
            f"rho size {rho.size} does not match dominance size {dominance.size}."
        )
    expected_uniform = 1.0 / max(float(rho.size), 1.0)
    return rho * (dominance / (expected_uniform + eps))


def safe_corrcoef(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation with a robust zero-variance fallback."""

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        return 0.0
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
