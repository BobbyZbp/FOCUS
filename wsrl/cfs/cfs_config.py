"""Configuration constants for CFS-D.

CFS-D = Conservatism Footprint + Target Dominance.
The module is intentionally small: all CFS-specific behavior lives under
``wsrl/cfs`` and the original SAC/finetune code only calls thin hooks.
"""

from __future__ import annotations

CFS_MODES = (
    "low_eta",  # main CFS-D intervention
    "low_rho",  # original CFS-lite ablation
    "high_eta",  # negative/control ablation
    "random_topk",  # random top-k control
)

DEFAULT_CFS_MODE = "low_eta"
DEFAULT_CFS_TOP_K = 5
DEFAULT_CFS_CALIB_N = 50_000
DEFAULT_CFS_E_WEIGHT = 0.1
DEFAULT_CFS_MIN_CV = 0.05
DEFAULT_CFS_DOMINANCE_SAMPLES = 20_000


def normalize_cfs_mode(mode: str) -> str:
    """Validate and normalize a CFS selection mode."""

    mode = str(mode).strip().lower()
    if mode not in CFS_MODES:
        raise ValueError(f"Unknown CFS mode {mode!r}; expected one of {CFS_MODES}.")
    return mode
