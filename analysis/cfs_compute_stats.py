"""Compute CFS-D diagnostics for a pretrained REDQ checkpoint.

Example:
    python analysis/cfs_compute_stats.py \
      --env antmaze-medium-diverse-v2 \
      --checkpoint /path/to/checkpoint \
      --agent calql \
      --use_redq \
      --config experiments/configs/train_config.py:antmaze_cql \
      --num_samples 50000 \
      --output results/cfs/cfs_stats_medium_seed0.csv
"""

from __future__ import annotations

import json
import os

import gym
import jax
import numpy as np
from absl import app, flags, logging
from flax.training import checkpoints
from ml_collections import config_flags

from experiments.configs.ensemble_config import add_redq_config
from wsrl.agents import agents
from wsrl.cfs.cfs_head_selection import select_head_pool
from wsrl.cfs.cfs_stats import (
    compute_cfs_statistics,
    selected_by_default_modes,
    write_cfs_stats_csv,
)
from wsrl.envs.adroit_binary_dataset import get_hand_dataset_with_mc_calculation
from wsrl.envs.d4rl_dataset import (
    get_d4rl_dataset,
    get_d4rl_dataset_with_mc_calculation,
)
from wsrl.envs.env_common import get_env_type, make_gym_env
from wsrl.utils.train_utils import subsample_batch

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "antmaze-large-diverse-v2", "Environment to use.")
flags.DEFINE_string("checkpoint", "", "Checkpoint path to inspect.")
flags.DEFINE_string("agent", "calql", "Agent type used to restore the checkpoint.")
flags.DEFINE_bool(
    "use_redq", False, "Apply REDQ ensemble config before constructing agent."
)
flags.DEFINE_float("reward_scale", 10.0, "Reward scale.")
flags.DEFINE_float("reward_bias", -5.0, "Reward bias.")
flags.DEFINE_float("clip_action", 0.99999, "Action clip limit.")
flags.DEFINE_integer("batch_size", 1024, "Agent construction batch size.")
flags.DEFINE_integer("cfs_batch_size", 4096, "CFS forward-pass batch size.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer(
    "num_samples", 50_000, "Number of offline samples for CFS calibration."
)
flags.DEFINE_float("cfs_e_weight", 0.1, "Bellman inconsistency weight in rho_h.")
flags.DEFINE_integer("cfs_top_k", 5, "Number of CFS heads to select.")
flags.DEFINE_enum(
    "cfs_mode",
    "low_eta",
    ["low_eta", "low_rho", "high_eta", "random_topk"],
    "Selection mode used for the 'selected' CSV column.",
)
flags.DEFINE_integer(
    "cfs_dominance_samples",
    20_000,
    "Monte-Carlo dominance samples if exact REDQ subset enumeration is too large.",
)
flags.DEFINE_string("output", "results/cfs/cfs_stats.csv", "CSV output path.")
config_flags.DEFINE_config_file(
    "config",
    None,
    "Training hyperparameter config, e.g. experiments/configs/train_config.py:antmaze_cql.",
    lock_config=False,
)


def _load_dataset(env_type: str):
    if env_type == "adroit-binary":
        return get_hand_dataset_with_mc_calculation(
            FLAGS.env,
            gamma=FLAGS.config.agent_kwargs.discount,
            reward_scale=FLAGS.reward_scale,
            reward_bias=FLAGS.reward_bias,
            clip_action=FLAGS.clip_action,
        )
    if FLAGS.agent == "calql":
        return get_d4rl_dataset_with_mc_calculation(
            FLAGS.env,
            reward_scale=FLAGS.reward_scale,
            reward_bias=FLAGS.reward_bias,
            clip_action=FLAGS.clip_action,
            gamma=FLAGS.config.agent_kwargs.discount,
        )
    return get_d4rl_dataset(
        FLAGS.env,
        reward_scale=FLAGS.reward_scale,
        reward_bias=FLAGS.reward_bias,
        clip_action=FLAGS.clip_action,
    )


def main(_):
    if FLAGS.config is None:
        raise ValueError("--config is required.")
    if FLAGS.checkpoint == "":
        raise ValueError("--checkpoint is required.")
    if not os.path.exists(FLAGS.checkpoint):
        raise FileNotFoundError(FLAGS.checkpoint)

    if FLAGS.use_redq:
        FLAGS.config.agent_kwargs = add_redq_config(FLAGS.config.agent_kwargs)

    env_type = get_env_type(FLAGS.env)
    env = make_gym_env(
        env_name=FLAGS.env,
        reward_scale=FLAGS.reward_scale,
        reward_bias=FLAGS.reward_bias,
        scale_and_clip_action=env_type in ("antmaze", "kitchen", "locomotion"),
        action_clip_lim=FLAGS.clip_action,
        seed=FLAGS.seed,
    )
    dataset = _load_dataset(env_type)

    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, construct_rng = jax.random.split(rng)
    example_batch = subsample_batch(dataset, FLAGS.batch_size)
    agent = agents[FLAGS.agent].create(
        rng=construct_rng,
        observations=example_batch["observations"],
        actions=example_batch["actions"],
        encoder_def=None,
        **FLAGS.config.agent_kwargs,
    )
    agent = checkpoints.restore_checkpoint(FLAGS.checkpoint, target=agent)

    critic_subsample_size = int(
        agent.config.get(
            "critic_subsample_size",
            FLAGS.config.agent_kwargs.get("critic_subsample_size", 2),
        )
        or 2
    )

    stats = compute_cfs_statistics(
        agent=agent,
        dataset=dataset,
        gamma=float(FLAGS.config.agent_kwargs.discount),
        num_samples=int(FLAGS.num_samples),
        batch_size=int(FLAGS.cfs_batch_size),
        e_weight=float(FLAGS.cfs_e_weight),
        critic_subsample_size=critic_subsample_size,
        dominance_samples=int(FLAGS.cfs_dominance_samples),
        seed=int(FLAGS.seed),
    )

    selected = select_head_pool(
        rho=stats.rho,
        eta=stats.eta,
        mode=FLAGS.cfs_mode,
        top_k=int(FLAGS.cfs_top_k),
        seed=int(FLAGS.seed),
    )
    selected_modes = selected_by_default_modes(
        stats, top_k=int(FLAGS.cfs_top_k), seed=int(FLAGS.seed)
    )
    write_cfs_stats_csv(
        FLAGS.output,
        stats,
        selected_heads=selected,
        selected_by_mode=selected_modes,
    )

    summary = stats.summary(selected_heads=selected)
    summary.update(
        {
            "cfs/output": FLAGS.output,
            "cfs/mode": FLAGS.cfs_mode,
            "cfs/top_k": int(FLAGS.cfs_top_k),
            "cfs/head_pool": ",".join(map(str, selected)),
        }
    )
    logging.info("CFS summary: %s", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    app.run(main)
