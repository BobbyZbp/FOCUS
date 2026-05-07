#!/usr/bin/env python3
"""Apply the small CFS-D hooks to an ST-CCQ checkout.

Run from the ST-CCQ repository root after copying/unzipping the new CFS files:

    python apply_cfs_patch.py

The script only edits:
  - finetune.py
  - wsrl/agents/sac.py

All CFS-specific implementation lives in new files under wsrl/cfs and analysis/.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()


def _read(path: str) -> str:
    p = ROOT / path
    if not p.exists():
        raise FileNotFoundError(
            f"Could not find {path}; run from the ST-CCQ repo root."
        )
    return p.read_text()


def _write(path: str, text: str) -> None:
    (ROOT / path).write_text(text)


def _insert_after(
    text: str, anchor: str, insertion: str, *, marker: str, path: str
) -> str:
    if marker in text:
        return text
    if anchor not in text:
        raise RuntimeError(f"Anchor not found in {path}: {anchor!r}")
    return text.replace(anchor, anchor + insertion, 1)


def _insert_before(
    text: str, anchor: str, insertion: str, *, marker: str, path: str
) -> str:
    if marker in text:
        return text
    if anchor not in text:
        raise RuntimeError(f"Anchor not found in {path}: {anchor!r}")
    return text.replace(anchor, insertion + anchor, 1)


def _replace_once(text: str, old: str, new: str, *, marker: str, path: str) -> str:
    if marker in text:
        return text
    if old not in text:
        raise RuntimeError(
            f"Block to replace not found in {path}. Look for nearby changed formatting."
        )
    return text.replace(old, new, 1)


def patch_finetune() -> None:
    path = "finetune.py"
    text = _read(path)

    text = _insert_after(
        text,
        "from wsrl.agents.btccq import BTCCQAgent\n",
        "from wsrl.cfs.cfs_calibration import maybe_apply_cfs_calibration\n",
        marker="maybe_apply_cfs_calibration",
        path=path,
    )

    cfs_flags = """\n# CFS-D flags (ignored unless --use_cfs is set)\nflags.DEFINE_bool("use_cfs", False, "Use CFS-D target-head selection for REDQ online targets.")\nflags.DEFINE_enum(\n    "cfs_mode",\n    "low_eta",\n    ["low_eta", "low_rho", "high_eta", "random_topk"],\n    "CFS head selection mode.",\n)\nflags.DEFINE_integer("cfs_top_k", 5, "Number of selected CFS REDQ heads.")\nflags.DEFINE_integer("cfs_calib_n", 50_000, "Number of offline samples for CFS calibration.")\nflags.DEFINE_integer("cfs_calib_batch_size", 4096, "Forward-pass batch size for CFS calibration.")\nflags.DEFINE_float("cfs_e_weight", 0.1, "Bellman error weight in the CFS rho score.")\nflags.DEFINE_integer(\n    "cfs_dominance_samples",\n    20_000,\n    "Monte-Carlo dominance samples if exact REDQ subset enumeration is too large.",\n)\nflags.DEFINE_float("cfs_min_cv", 0.05, "Minimum rho CV used by --cfs_require_heterogeneity.")\nflags.DEFINE_bool(\n    "cfs_require_heterogeneity",\n    False,\n    "Disable CFS online intervention if rho_cv < cfs_min_cv.",\n)\nflags.DEFINE_string(\n    "cfs_stats_output",\n    "",\n    "Optional CSV path for CFS calibration stats during finetune.",\n)\n"""
    text = _insert_before(
        text,
        "config_flags.DEFINE_config_file(\n",
        cfs_flags,
        marker="cfs_calib_n",
        path=path,
    )

    cfs_calibration_block = """            # CFS-D: at the offline -> online boundary, score the frozen\n            # REDQ heads and optionally restrict online REDQ target-head\n            # sampling to the selected low-footprint / low-influence pool.\n            if FLAGS.use_cfs:\n                logging.info("CFS-D: running transition-time calibration...")\n                critic_subsample_size = FLAGS.config.agent_kwargs.get(\n                    "critic_subsample_size", 2\n                )\n                agent, cfs_info = maybe_apply_cfs_calibration(\n                    agent=agent,\n                    dataset=dataset,\n                    gamma=FLAGS.config.agent_kwargs.discount,\n                    use_cfs=True,\n                    mode=FLAGS.cfs_mode,\n                    top_k=FLAGS.cfs_top_k,\n                    num_samples=FLAGS.cfs_calib_n,\n                    batch_size=FLAGS.cfs_calib_batch_size,\n                    e_weight=FLAGS.cfs_e_weight,\n                    critic_subsample_size=critic_subsample_size,\n                    dominance_samples=FLAGS.cfs_dominance_samples,\n                    min_cv=FLAGS.cfs_min_cv,\n                    require_heterogeneity=FLAGS.cfs_require_heterogeneity,\n                    output_path=FLAGS.cfs_stats_output,\n                    seed=FLAGS.seed,\n                )\n                logging.info("CFS-D calibration: %s", cfs_info)\n                wandb_logger.log({"cfs_calibration": cfs_info}, step=step)\n\n"""
    text = _insert_before(
        text,
        "            # BT-CCQ: at the offline -> online boundary, freeze the\n",
        cfs_calibration_block,
        marker="CFS-D: running transition-time calibration",
        path=path,
    )

    _write(path, text)


def patch_sac() -> None:
    path = "wsrl/agents/sac.py"
    text = _read(path)

    text = _insert_after(
        text,
        "from wsrl.networks.mlp import MLP\n",
        "from wsrl.cfs.cfs_head_selection import sample_redq_target_heads\n",
        marker="sample_redq_target_heads",
        path=path,
    )

    old_sample = """            subsample_idcs = jax.random.randint(\n                subsample_key,\n                (self.config["critic_subsample_size"],),\n                0,\n                self.config["critic_ensemble_size"],\n            )\n"""
    new_sample = """            subsample_idcs = sample_redq_target_heads(\n                key=subsample_key,\n                critic_subsample_size=self.config["critic_subsample_size"],\n                critic_ensemble_size=self.config["critic_ensemble_size"],\n                config=self.config,\n            )\n"""
    text = _replace_once(
        text,
        old_sample,
        new_sample,
        marker="subsample_idcs = sample_redq_target_heads(",
        path=path,
    )

    old_update_config = '''    def update_config(self, new_config):\n        """update the frozen self.config"""\n        object.__setattr__(self, "config", self.config.copy(new_config))\n'''
    new_update_config = '''    def update_config(self, new_config):\n        """update the frozen self.config"""\n        updated_config = dict(self.config)\n        updated_config.update(dict(new_config))\n        object.__setattr__(self, "config", updated_config)\n'''
    text = _replace_once(
        text,
        old_update_config,
        new_update_config,
        marker="updated_config = dict(self.config)",
        path=path,
    )

    _write(path, text)


def main() -> None:
    patch_finetune()
    patch_sac()
    print("CFS-D hooks applied to finetune.py and wsrl/agents/sac.py")


if __name__ == "__main__":
    main()
