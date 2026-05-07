"""Online-transition CFS-D calibration hook."""

from __future__ import annotations

from typing import Mapping, Sequence

import jax

from wsrl.cfs.cfs_config import DEFAULT_CFS_MIN_CV, normalize_cfs_mode
from wsrl.cfs.cfs_head_selection import select_head_pool, validate_head_pool
from wsrl.cfs.cfs_stats import (
    compute_cfs_statistics,
    selected_by_default_modes,
    write_cfs_stats_csv,
)


def _safe_update_agent_config(agent, updates: Mapping) -> None:
    """Update agent.config without depending on a specific dict/FrozenDict type."""

    if hasattr(agent, "update_config"):
        try:
            agent.update_config(dict(updates))
            return
        except TypeError:
            # Some versions used dict.copy(new_config), which fails for a plain dict.
            pass

    config = dict(getattr(agent, "config", {}))
    config.update(dict(updates))
    object.__setattr__(agent, "config", config)


def _config_get(config: Mapping, key: str, default=None):
    try:
        return config.get(key, default)
    except AttributeError:
        return default


def maybe_apply_cfs_calibration(
    *,
    agent,
    dataset,
    gamma: float,
    use_cfs: bool = True,
    mode: str = "low_eta",
    top_k: int = 5,
    num_samples: int = 50_000,
    batch_size: int = 4096,
    e_weight: float = 0.1,
    critic_subsample_size: int | None = None,
    dominance_samples: int = 20_000,
    min_cv: float = DEFAULT_CFS_MIN_CV,
    require_heterogeneity: bool = False,
    output_path: str = "",
    seed: int = 0,
) -> tuple[object, dict]:
    """Calibrate CFS at the offline-to-online transition.

    Returns the same agent object with ``agent.config`` updated in-place.
    """

    if not use_cfs:
        return agent, {"cfs/enabled": 0}

    mode = normalize_cfs_mode(mode)
    agent_config = getattr(agent, "config", {})
    ensemble_size = int(_config_get(agent_config, "critic_ensemble_size", 1))
    if ensemble_size <= 1:
        _safe_update_agent_config(agent, {"use_cfs": False})
        return agent, {
            "cfs/enabled": 0,
            "cfs/disabled_reason": "critic_ensemble_size<=1",
            "cfs/critic_ensemble_size": ensemble_size,
        }

    if critic_subsample_size is None:
        critic_subsample_size = _config_get(agent_config, "critic_subsample_size", 2)
    critic_subsample_size = int(critic_subsample_size or 2)

    stats = compute_cfs_statistics(
        agent=agent,
        dataset=dataset,
        gamma=float(gamma),
        num_samples=int(num_samples),
        batch_size=int(batch_size),
        e_weight=float(e_weight),
        critic_subsample_size=critic_subsample_size,
        dominance_samples=int(dominance_samples),
        seed=int(seed),
    )

    all_modes = selected_by_default_modes(stats, top_k=int(top_k), seed=int(seed))
    selected_heads = select_head_pool(
        rho=stats.rho,
        eta=stats.eta,
        mode=mode,
        top_k=int(top_k),
        seed=int(seed),
    )
    selected_heads = validate_head_pool(
        selected_heads, critic_ensemble_size=ensemble_size
    )

    summary = stats.summary(selected_heads=selected_heads)
    summary.update(
        {
            "cfs/enabled": 1,
            "cfs/mode": mode,
            "cfs/top_k": int(min(int(top_k), ensemble_size)),
            "cfs/head_pool": ",".join(map(str, selected_heads)),
            "cfs/critic_ensemble_size": ensemble_size,
            "cfs/min_cv": float(min_cv),
            "cfs/require_heterogeneity": int(bool(require_heterogeneity)),
        }
    )

    if bool(require_heterogeneity) and float(stats.rho_cv) < float(min_cv):
        _safe_update_agent_config(
            agent,
            {
                "use_cfs": False,
                "cfs_disabled_reason": "rho_cv_below_min_cv",
                "cfs_rho_cv": float(stats.rho_cv),
            },
        )
        summary["cfs/enabled"] = 0
        summary["cfs/disabled_reason"] = "rho_cv_below_min_cv"
        if output_path:
            write_cfs_stats_csv(
                output_path,
                stats,
                selected_heads=selected_heads,
                selected_by_mode=all_modes,
            )
        return agent, summary

    _safe_update_agent_config(
        agent,
        {
            "use_cfs": True,
            "cfs_mode": mode,
            "cfs_head_pool": tuple(selected_heads),
            "cfs_top_k": int(min(int(top_k), ensemble_size)),
            "cfs_rho_cv": float(stats.rho_cv),
            "cfs_rho_d_corr": float(stats.rho_d_corr),
        },
    )

    if output_path:
        write_cfs_stats_csv(
            output_path,
            stats,
            selected_heads=selected_heads,
            selected_by_mode=all_modes,
        )

    return agent, summary
