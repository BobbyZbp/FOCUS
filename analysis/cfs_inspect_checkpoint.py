"""Inspect a checkpoint and verify that REDQ critic heads are restorable."""

from __future__ import annotations

import json
import os

import jax
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags

from experiments.configs.ensemble_config import add_redq_config
from wsrl.agents import agents
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
flags.DEFINE_integer("seed", 0, "Random seed.")
config_flags.DEFINE_config_file(
    "config",
    None,
    "Training hyperparameter config.",
    lock_config=False,
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
    if FLAGS.agent == "calql":
        dataset = get_d4rl_dataset_with_mc_calculation(
            FLAGS.env,
            reward_scale=FLAGS.reward_scale,
            reward_bias=FLAGS.reward_bias,
            clip_action=FLAGS.clip_action,
            gamma=FLAGS.config.agent_kwargs.discount,
        )
    else:
        dataset = get_d4rl_dataset(
            FLAGS.env,
            reward_scale=FLAGS.reward_scale,
            reward_bias=FLAGS.reward_bias,
            clip_action=FLAGS.clip_action,
        )

    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, construct_rng, critic_rng = jax.random.split(rng, 3)
    example_batch = subsample_batch(dataset, FLAGS.batch_size)
    agent = agents[FLAGS.agent].create(
        rng=construct_rng,
        observations=example_batch["observations"],
        actions=example_batch["actions"],
        encoder_def=None,
        **FLAGS.config.agent_kwargs,
    )
    agent = checkpoints.restore_checkpoint(FLAGS.checkpoint, target=agent)

    q = agent.forward_critic(
        example_batch["observations"][:8],
        example_batch["actions"][:8],
        rng=critic_rng,
        train=False,
    )
    info = {
        "checkpoint": FLAGS.checkpoint,
        "env": FLAGS.env,
        "agent": FLAGS.agent,
        "critic_ensemble_size_config": int(
            agent.config.get("critic_ensemble_size", -1)
        ),
        "critic_subsample_size_config": agent.config.get("critic_subsample_size", None),
        "forward_critic_shape": tuple(int(x) for x in q.shape),
        "state_step": int(agent.state.step),
    }
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    app.run(main)
