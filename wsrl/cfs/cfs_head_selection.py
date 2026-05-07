"""Head-selection and REDQ sampling helpers for CFS-D."""

from __future__ import annotations

from typing import Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from wsrl.cfs.cfs_config import normalize_cfs_mode


def _as_1d_numpy(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array; got shape {arr.shape}.")
    return arr


def validate_head_pool(
    head_pool: Iterable[int], critic_ensemble_size: int
) -> tuple[int, ...]:
    """Return a sorted-free tuple after checking that heads are valid."""

    pool = tuple(int(h) for h in head_pool)
    if len(pool) == 0:
        raise ValueError("CFS head pool is empty.")
    if len(set(pool)) != len(pool):
        raise ValueError(f"CFS head pool has duplicate heads: {pool}.")
    bad = [h for h in pool if h < 0 or h >= int(critic_ensemble_size)]
    if bad:
        raise ValueError(
            f"CFS head pool contains invalid heads {bad}; "
            f"critic_ensemble_size={critic_ensemble_size}."
        )
    return pool


def select_head_pool(
    *,
    rho: Sequence[float],
    eta: Sequence[float] | None = None,
    mode: str = "low_eta",
    top_k: int = 5,
    seed: int = 0,
) -> tuple[int, ...]:
    """Select a REDQ target-head pool from CFS scores.

    Args:
        rho: Per-head conservatism footprint score.
        eta: Per-head target-dominant conservatism influence score.
        mode: One of ``low_eta``, ``low_rho``, ``high_eta``, ``random_topk``.
        top_k: Number of heads to keep.
        seed: RNG seed used only by ``random_topk``.

    Returns:
        Tuple of selected head ids.
    """

    mode = normalize_cfs_mode(mode)
    rho_arr = _as_1d_numpy(rho, "rho")
    h = rho_arr.size
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}.")
    top_k = min(top_k, h)

    if mode == "low_rho":
        order = np.argsort(rho_arr)
    elif mode in ("low_eta", "high_eta"):
        if eta is None:
            raise ValueError(f"eta is required for CFS mode {mode!r}.")
        eta_arr = _as_1d_numpy(eta, "eta")
        if eta_arr.size != h:
            raise ValueError(f"eta size {eta_arr.size} does not match rho size {h}.")
        order = np.argsort(eta_arr)
        if mode == "high_eta":
            order = order[::-1]
    elif mode == "random_topk":
        rng = np.random.default_rng(seed)
        order = rng.permutation(h)
    else:  # normalize_cfs_mode should prevent this
        raise AssertionError(mode)

    return tuple(int(x) for x in order[:top_k])


def sample_redq_target_heads(
    *,
    key: jax.Array,
    critic_subsample_size: int,
    critic_ensemble_size: int,
    config: dict,
) -> jax.Array:
    """Sample REDQ target heads, optionally restricted by CFS.

    This helper is designed to be called from ``SACAgent.critic_loss_fn``.
    It preserves the original REDQ behavior when ``config['use_cfs']`` is false.

    REDQ samples with replacement in the current ST-CCQ implementation via
    ``jax.random.randint``; CFS keeps that behavior but changes the support from
    all heads to the calibrated CFS head pool.
    """

    if critic_subsample_size is None:
        raise ValueError("critic_subsample_size must not be None when sampling heads.")

    if bool(config.get("use_cfs", False)):
        pool = jnp.asarray(config["cfs_head_pool"], dtype=jnp.int32)
        pool_idx = jax.random.randint(
            key,
            (int(critic_subsample_size),),
            0,
            pool.shape[0],
        )
        return pool[pool_idx]

    return jax.random.randint(
        key,
        (int(critic_subsample_size),),
        0,
        int(critic_ensemble_size),
    )
