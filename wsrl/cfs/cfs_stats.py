"""CFS-D diagnostic statistics.

The main quantities are:

  p_h   = E[(z_off - Q_h(s,a))_+]
  e_h   = E[(Q_h(s,a) - z_off)^2]
  rho_h = p_h + lambda_e * e_h
  d_h   = REDQ min-target win rate
  eta_h = rho_h * (d_h / (1/H))
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Mapping, Sequence

import jax
import numpy as np

from wsrl.cfs.cfs_head_selection import select_head_pool
from wsrl.cfs.cfs_target_dominance import (
    compute_eta,
    estimate_min_win_rate,
    safe_corrcoef,
)


@dataclass(frozen=True)
class CFSStats:
    head_id: np.ndarray
    pess_gap: np.ndarray
    bellman_err: np.ndarray
    rho: np.ndarray
    dominance: np.ndarray
    eta: np.ndarray
    q_sa_mean: np.ndarray
    q_next_mean: np.ndarray
    rho_mean: float
    rho_std: float
    rho_cv: float
    rho_d_corr: float
    num_samples: int
    critic_subsample_size: int
    e_weight: float

    def summary(
        self, selected_heads: Sequence[int] | None = None, prefix: str = "cfs"
    ) -> dict:
        selected_heads = tuple(int(h) for h in (selected_heads or ()))
        return {
            f"{prefix}/rho_mean": self.rho_mean,
            f"{prefix}/rho_std": self.rho_std,
            f"{prefix}/rho_cv": self.rho_cv,
            f"{prefix}/rho_d_corr": self.rho_d_corr,
            f"{prefix}/eta_mean": float(np.mean(self.eta)),
            f"{prefix}/eta_std": float(np.std(self.eta)),
            f"{prefix}/dominance_entropy": float(
                -np.sum(self.dominance * np.log(self.dominance + 1e-12))
            ),
            f"{prefix}/num_samples": int(self.num_samples),
            f"{prefix}/critic_subsample_size": int(self.critic_subsample_size),
            f"{prefix}/selected_heads": ",".join(map(str, selected_heads)),
        }


def _require_dataset_keys(dataset: Mapping[str, np.ndarray]) -> None:
    required = ("observations", "actions", "next_observations", "rewards")
    missing = [k for k in required if k not in dataset]
    if missing:
        raise KeyError(f"Dataset is missing required CFS keys: {missing}.")
    if "masks" not in dataset and "dones" not in dataset and "terminals" not in dataset:
        raise KeyError("Dataset must contain one of: 'masks', 'dones', or 'terminals'.")


def _subsample_dataset(
    dataset: Mapping[str, np.ndarray], num_samples: int, seed: int
) -> dict:
    _require_dataset_keys(dataset)
    n = int(dataset["observations"].shape[0])
    if n <= 0:
        raise ValueError("Dataset is empty.")
    if num_samples is None or int(num_samples) <= 0:
        size = n
        replace = False
    else:
        size = int(num_samples)
        replace = size > n
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=size, replace=replace)
    keys = ["observations", "actions", "next_observations", "rewards"]
    for optional_key in ("masks", "dones", "terminals"):
        if optional_key in dataset:
            keys.append(optional_key)
    return {k: dataset[k][indices] for k in keys}


def _iter_slices(n: int, batch_size: int):
    for start in range(0, n, int(batch_size)):
        yield slice(start, min(start + int(batch_size), n))


def _slice_batch(batch: Mapping[str, np.ndarray], s: slice) -> dict:
    return {k: v[s] for k, v in batch.items()}


def _flatten_vector(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _ensure_head_batch(q: np.ndarray, name: str) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        q = q[None, :]
    if q.ndim != 2:
        raise ValueError(f"{name} must have shape (H, B); got {q.shape}.")
    return q


def _safe_target_params(agent):
    return getattr(agent.state, "target_params", agent.state.params)


def compute_cfs_statistics(
    *,
    agent,
    dataset: Mapping[str, np.ndarray],
    gamma: float,
    num_samples: int = 50_000,
    batch_size: int = 4096,
    e_weight: float = 0.1,
    critic_subsample_size: int = 2,
    dominance_samples: int = 20_000,
    seed: int = 0,
) -> CFSStats:
    """Compute CFS-D statistics for a pretrained REDQ checkpoint."""

    calib = _subsample_dataset(dataset, num_samples=num_samples, seed=seed)
    n = int(calib["rewards"].shape[0])
    if n <= 0:
        raise ValueError("No calibration samples available for CFS.")

    apply_fn = agent.state.apply_fn
    params = agent.state.params
    target_params = _safe_target_params(agent)
    rng = jax.random.PRNGKey(seed)

    p_sum = None
    e_sum = None
    q_sa_sum = None
    q_next_sum = None
    q_next_chunks = []
    total = 0

    for sl in _iter_slices(n, batch_size):
        mini = _slice_batch(calib, sl)
        obs = mini["observations"]
        actions = mini["actions"]
        next_obs = mini["next_observations"]
        rewards = _flatten_vector(mini["rewards"])
        if "masks" in mini:
            masks = _flatten_vector(mini["masks"])
        elif "dones" in mini:
            masks = 1.0 - _flatten_vector(mini["dones"])
        else:
            masks = 1.0 - _flatten_vector(mini["terminals"])

        rng, actor_key, q_sa_key, q_next_key = jax.random.split(rng, 4)

        # Frozen offline policy action at s'.
        next_action_dist = apply_fn(
            {"params": params},
            next_obs,
            name="actor",
            rngs={"dropout": actor_key},
            train=False,
        )
        next_actions = next_action_dist.mode()

        # Per-head Q_h(s, a) from live/frozen params.
        q_sa = apply_fn(
            {"params": params},
            obs,
            actions,
            name="critic",
            rngs={"dropout": q_sa_key},
            train=False,
        )
        q_sa = _ensure_head_batch(jax.device_get(q_sa), "q_sa")

        # Per-head target Q_j(s', pi_off(s')) from target params, matching REDQ backup.
        q_next = apply_fn(
            {"params": target_params},
            next_obs,
            next_actions,
            name="critic",
            rngs={"dropout": q_next_key},
            train=False,
        )
        q_next = _ensure_head_batch(jax.device_get(q_next), "q_next")

        z_off = rewards + float(gamma) * masks * np.min(q_next, axis=0)
        diff = z_off[None, :] - q_sa

        batch_p = np.maximum(diff, 0.0).sum(axis=1)
        batch_e = np.square(q_sa - z_off[None, :]).sum(axis=1)
        batch_q_sa_sum = q_sa.sum(axis=1)
        batch_q_next_sum = q_next.sum(axis=1)

        if p_sum is None:
            h = q_sa.shape[0]
            p_sum = np.zeros(h, dtype=np.float64)
            e_sum = np.zeros(h, dtype=np.float64)
            q_sa_sum = np.zeros(h, dtype=np.float64)
            q_next_sum = np.zeros(h, dtype=np.float64)
        elif q_sa.shape[0] != p_sum.shape[0]:
            raise ValueError("Critic ensemble size changed across batches.")

        p_sum += batch_p
        e_sum += batch_e
        q_sa_sum += batch_q_sa_sum
        q_next_sum += batch_q_next_sum
        q_next_chunks.append(q_next)
        total += q_sa.shape[1]

    if p_sum is None or total == 0:
        raise RuntimeError("CFS failed to accumulate any statistics.")

    pess_gap = p_sum / float(total)
    bellman_err = e_sum / float(total)
    rho = pess_gap + float(e_weight) * bellman_err
    q_sa_mean = q_sa_sum / float(total)
    q_next_mean = q_next_sum / float(total)

    q_next_all = np.concatenate(q_next_chunks, axis=1)
    dominance = estimate_min_win_rate(
        q_next_all,
        critic_subsample_size=int(critic_subsample_size),
        num_samples=int(dominance_samples),
        seed=seed,
    )
    eta = compute_eta(rho, dominance)
    rho_mean = float(np.mean(rho))
    rho_std = float(np.std(rho))
    rho_cv = float(rho_std / (rho_mean + 1e-8))
    rho_d_corr = safe_corrcoef(rho, dominance)

    return CFSStats(
        head_id=np.arange(rho.shape[0], dtype=np.int64),
        pess_gap=pess_gap,
        bellman_err=bellman_err,
        rho=rho,
        dominance=dominance,
        eta=eta,
        q_sa_mean=q_sa_mean,
        q_next_mean=q_next_mean,
        rho_mean=rho_mean,
        rho_std=rho_std,
        rho_cv=rho_cv,
        rho_d_corr=rho_d_corr,
        num_samples=int(total),
        critic_subsample_size=int(critic_subsample_size),
        e_weight=float(e_weight),
    )


def selected_by_default_modes(
    stats: CFSStats, top_k: int, seed: int = 0
) -> dict[str, tuple[int, ...]]:
    """Compute the standard CFS selection/control pools for CSV annotation."""

    return {
        "low_rho": select_head_pool(
            rho=stats.rho, eta=stats.eta, mode="low_rho", top_k=top_k, seed=seed
        ),
        "low_eta": select_head_pool(
            rho=stats.rho, eta=stats.eta, mode="low_eta", top_k=top_k, seed=seed
        ),
        "high_eta": select_head_pool(
            rho=stats.rho, eta=stats.eta, mode="high_eta", top_k=top_k, seed=seed
        ),
        "random_topk": select_head_pool(
            rho=stats.rho, eta=stats.eta, mode="random_topk", top_k=top_k, seed=seed
        ),
    }


def write_cfs_stats_csv(
    path: str,
    stats: CFSStats,
    *,
    selected_heads: Sequence[int] | None = None,
    selected_by_mode: Mapping[str, Sequence[int]] | None = None,
) -> None:
    """Write one row per critic head with global CFS metadata repeated."""

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    selected_heads = set(int(h) for h in (selected_heads or ()))
    selected_by_mode = selected_by_mode or {}
    mode_sets = {
        mode: set(int(h) for h in heads) for mode, heads in selected_by_mode.items()
    }

    rows = []
    for i, h in enumerate(stats.head_id):
        row = {
            "head_id": int(h),
            "pess_gap_p_h": float(stats.pess_gap[i]),
            "bellman_err_e_h": float(stats.bellman_err[i]),
            "rho_h": float(stats.rho[i]),
            "min_win_rate_d_h": float(stats.dominance[i]),
            "eta_h": float(stats.eta[i]),
            "q_sa_mean_h": float(stats.q_sa_mean[i]),
            "q_next_mean_h": float(stats.q_next_mean[i]),
            "selected": int(int(h) in selected_heads),
            "rho_mean": stats.rho_mean,
            "rho_std": stats.rho_std,
            "rho_cv": stats.rho_cv,
            "rho_d_corr": stats.rho_d_corr,
            "num_samples": stats.num_samples,
            "critic_subsample_size": stats.critic_subsample_size,
            "e_weight": stats.e_weight,
        }
        for mode, heads in mode_sets.items():
            row[f"selected_{mode}"] = int(int(h) in heads)
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
