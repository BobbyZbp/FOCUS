import os
from functools import partial

import gym
import jax
import numpy as np
import tqdm
from absl import app, flags, logging
from flax.training import checkpoints
from ml_collections import config_flags

from experiments.configs.ensemble_config import add_redq_config
from wsrl.agents import agents
from wsrl.agents.btccq import BTCCQAgent
from wsrl.cfs.cfs_calibration import maybe_apply_cfs_calibration
from wsrl.common.evaluation import evaluate_with_trajectories
from wsrl.common.wandb import WandBLogger
from wsrl.data.replay_buffer import ReplayBuffer, ReplayBufferMC
from wsrl.envs.adroit_binary_dataset import get_hand_dataset_with_mc_calculation
from wsrl.envs.d4rl_dataset import (
    get_d4rl_dataset,
    get_d4rl_dataset_with_mc_calculation,
)
from wsrl.envs.env_common import get_env_type, make_gym_env
from wsrl.utils.timer_utils import Timer
from wsrl.utils.train_utils import concatenate_batches, subsample_batch

FLAGS = flags.FLAGS

# env
flags.DEFINE_string("env", "antmaze-large-diverse-v2", "Environemnt to use")
flags.DEFINE_float("reward_scale", 1.0, "Reward scale.")
flags.DEFINE_float("reward_bias", -1.0, "Reward bias.")
flags.DEFINE_float(
    "clip_action",
    0.99999,
    "Clip actions to be between [-n, n]. This is needed for tanh policies.",
)

# training
flags.DEFINE_integer("num_offline_steps", 1_000_000, "Number of offline epochs.")
flags.DEFINE_integer("num_online_steps", 500_000, "Number of online epochs.")
flags.DEFINE_float(
    "offline_data_ratio",
    0.0,
    "How much offline data to retain in each online batch update",
)
flags.DEFINE_string(
    "online_sampling_method",
    "mixed",
    """Method of sampling data during online update: mixed or append.
    `mixed` samples from a mix of offline and online data according to offline_data_ratio.
    `append` adds offline data to replay buffer and samples from it.""",
)
flags.DEFINE_bool(
    "online_use_cql_loss",
    True,
    """When agent is CQL/CalQL, whether to use CQL loss for the online phase (use SAC loss if False)""",
)
flags.DEFINE_integer(
    "warmup_steps", 0, "number of warmup steps (WSRL) before performing online updates"
)

# agent
flags.DEFINE_string("agent", "calql", "what RL agent to use")
flags.DEFINE_integer("utd", 1, "update-to-data ratio of the critic")
flags.DEFINE_integer("batch_size", 256, "batch size for training")
flags.DEFINE_integer("replay_buffer_capacity", int(2e6), "Replay buffer capacity")
flags.DEFINE_bool("use_redq", False, "Use an ensemble of Q-functions for the agent")

# experiment house keeping
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string(
    "save_dir",
    os.path.expanduser("~/wsrl_log"),
    "Directory to save the logs and checkpoints",
)
flags.DEFINE_string("resume_path", "", "Path to resume from")
flags.DEFINE_integer("log_interval", 5_000, "Log every n steps")
flags.DEFINE_integer("eval_interval", 20_000, "Evaluate every n steps")
flags.DEFINE_integer("save_interval", 100_000, "Save every n steps.")
flags.DEFINE_integer(
    "n_eval_trajs", 20, "Number of trajectories to use for each evaluation."
)
flags.DEFINE_bool("deterministic_eval", True, "Whether to use deterministic evaluation")

# wandb
flags.DEFINE_string("exp_name", "", "Experiment name for wandb logging")
flags.DEFINE_string("project", None, "Wandb project folder")
flags.DEFINE_string("group", None, "Wandb group of the experiment")
flags.DEFINE_bool("debug", False, "If true, no logging to wandb")

# BT-CCQ flags (ignored when agent != btccq)
flags.DEFINE_float("btccq_alpha", 0.1, "BT-CCQ miscoverage level for q_hat quantile")
flags.DEFINE_float("btccq_w_out", 0.2, "BT-CCQ minimum gate weight for OOD transitions")
flags.DEFINE_float("btccq_eps", 1e-6, "BT-CCQ numerical stability constant")
flags.DEFINE_float(
    "btccq_calib_ratio",
    0.1,
    "Fraction of offline dataset to hold out for q_hat calibration",
)


# CFS-D flags (ignored unless --use_cfs is set)
flags.DEFINE_bool(
    "use_cfs", False, "Use CFS-D target-head selection for REDQ online targets."
)
flags.DEFINE_enum(
    "cfs_mode",
    "low_eta",
    ["low_eta", "low_rho", "high_eta", "random_topk"],
    "CFS head selection mode.",
)
flags.DEFINE_integer("cfs_top_k", 5, "Number of selected CFS REDQ heads.")
flags.DEFINE_integer(
    "cfs_calib_n", 50_000, "Number of offline samples for CFS calibration."
)
flags.DEFINE_integer(
    "cfs_calib_batch_size", 4096, "Forward-pass batch size for CFS calibration."
)
flags.DEFINE_float("cfs_e_weight", 0.1, "Bellman error weight in the CFS rho score.")
flags.DEFINE_integer(
    "cfs_dominance_samples",
    20_000,
    "Monte-Carlo dominance samples if exact REDQ subset enumeration is too large.",
)
flags.DEFINE_float(
    "cfs_min_cv", 0.05, "Minimum rho CV used by --cfs_require_heterogeneity."
)
flags.DEFINE_bool(
    "cfs_require_heterogeneity",
    False,
    "Disable CFS online intervention if rho_cv < cfs_min_cv.",
)
flags.DEFINE_string(
    "cfs_stats_output",
    "",
    "Optional CSV path for CFS calibration stats during finetune.",
)
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def compute_fixed_q_stats(agent, fixed_diag_batch):
    """
    Compute Q-statistics on a fixed offline (s,a) batch.
    Stable distribution → suitable for Q-collapse plots across training.
    """
    rng = jax.random.PRNGKey(0)
    q = agent.forward_critic(
        fixed_diag_batch["observations"],
        fixed_diag_batch["actions"],
        rng=rng,
        train=False,
    )  # (ensemble_size, batch_size)
    q_min = q.min(axis=0)  # conservative estimate per (s,a)
    q_min = np.asarray(q_min)
    return {
        "q_diag/q_eval_mean": float(q_min.mean()),
        "q_diag/q_eval_std": float(q_min.std()),
        "q_diag/q_eval_min": float(q_min.min()),
        "q_diag/q_eval_max": float(q_min.max()),
        "q_diag/q_eval_p10": float(np.quantile(q_min, 0.10)),
        "q_diag/q_eval_p50": float(np.quantile(q_min, 0.50)),
        "q_diag/q_eval_p90": float(np.quantile(q_min, 0.90)),
    }


def calibrate_qhat(agent, dataset, gamma, alpha, calib_ratio, batch_size=4096):
    """
    Compute q_hat from a held-out calibration split of the offline dataset.

    e_i   = max(0, z_off - Q_off(s,a))
    z_off = r + gamma * masks * V_off(s')
    V_off = min_j Q_off^j(s', pi_off(s'))
    q_hat = quantile(e_i, 1-alpha)

    Uses the agent's current (frozen offline) params for all forward passes.
    No gradients are computed.
    """
    n = dataset["observations"].shape[0]
    rng = jax.random.PRNGKey(0)

    # Random calibration split (held-out from the rest of offline training)
    n_calib = int(n * calib_ratio)
    idx = np.random.permutation(n)[:n_calib]
    calib = {k: v[idx] for k, v in dataset.items()}

    offline_params = agent.state.params  # frozen at this point (before online updates)
    apply_fn = agent.state.apply_fn

    residuals = []
    for start in range(0, n_calib, batch_size):
        end = min(start + batch_size, n_calib)
        obs = calib["observations"][start:end]
        actions = calib["actions"][start:end]
        rewards = calib["rewards"][start:end]
        next_obs = calib["next_observations"][start:end]
        masks = calib["masks"][start:end]

        rng, k1, k2, k3 = jax.random.split(rng, 4)

        # Q_off(s, a)  →  (ensemble_size, batch)
        q_off_sa = apply_fn(
            {"params": offline_params},
            obs,
            actions,
            name="critic",
            rngs={"dropout": k1},
            train=False,
        ).min(
            axis=0
        )  # (batch,)

        # pi_off(s') — use mode (deterministic)
        a_next_off = apply_fn(
            {"params": offline_params},
            next_obs,
            name="actor",
            rngs={"dropout": k2},
            train=False,
        ).mode()

        # Q_off(s', pi_off(s'))  →  (ensemble_size, batch)
        v_off_next = apply_fn(
            {"params": offline_params},
            next_obs,
            a_next_off,
            name="critic",
            rngs={"dropout": k3},
            train=False,
        ).min(
            axis=0
        )  # (batch,)

        z_off = rewards + gamma * masks * v_off_next
        e_down = np.maximum(0.0, np.asarray(z_off) - np.asarray(q_off_sa))
        residuals.append(e_down.reshape(-1))

    residuals_oneside = np.concatenate(residuals)
    # `residuals_oneside` is the e_i = max(0, b_i) sequence. For the IQR-based
    # scale floor we also need the signed residual b_i = z_off - Q_off, which we
    # recover from the magnitude of e_i (it would have been simpler to also
    # accumulate b_i above, but keeping this self-contained for clarity).

    # NOTE: we re-walk the calibration set once more to also collect signed b_i.
    # This second pass is small (calib_ratio of dataset, no grad) and lets us
    # compute IQR(b) for the scale-aware floor without changing the loop above.
    signed_b = []
    for start in range(0, n_calib, batch_size):
        end = min(start + batch_size, n_calib)
        obs = calib["observations"][start:end]
        actions = calib["actions"][start:end]
        rewards = calib["rewards"][start:end]
        next_obs = calib["next_observations"][start:end]
        masks = calib["masks"][start:end]

        rng, k1, k2, k3 = jax.random.split(rng, 4)
        q_off_sa = apply_fn(
            {"params": offline_params},
            obs,
            actions,
            name="critic",
            rngs={"dropout": k1},
            train=False,
        ).min(axis=0)
        a_next_off = apply_fn(
            {"params": offline_params},
            next_obs,
            name="actor",
            rngs={"dropout": k2},
            train=False,
        ).mode()
        v_off_next = apply_fn(
            {"params": offline_params},
            next_obs,
            a_next_off,
            name="critic",
            rngs={"dropout": k3},
            train=False,
        ).min(axis=0)
        z_off = rewards + gamma * masks * v_off_next
        b = np.asarray(z_off) - np.asarray(q_off_sa)
        signed_b.append(b.reshape(-1))
    signed_b = np.concatenate(signed_b)

    # ---- Zero-inflation-aware calibration with scale-aware floor ----
    # Final BT-CCQ threshold:
    #     q_hat = max(q_tail, q_scale, q_min)
    # where
    #     q_tail  = Quantile_{1-alpha}({e_i : e_i > eps})        (positive tail)
    #     q_scale = lambda * IQR(b_i)   with b_i = z_off - Q_off (scale-aware floor)
    #     q_min   = small absolute floor                          (degenerate guard)
    #
    # Why three terms:
    # - q_tail handles dense and zero-inflated cases as long as enough positives.
    # - q_scale adapts to the natural noise level of the offline Bellman residual,
    #   so the threshold remains usable even when positives are scarce.
    # - q_min prevents pathological collapse to 0 in extreme degenerate regimes.
    eps_res = 1e-6
    min_positive_frac = 0.001  # need at least 0.1% positives to trust the tail
    lambda_scale = 0.05  # scale-aware floor coefficient (paper hyperparam)
    q_min = 1e-3  # absolute floor; never let q_hat collapse below this

    positive = residuals_oneside[residuals_oneside > eps_res]
    positive_frac = float(positive.size) / float(residuals_oneside.size)

    if positive.size >= max(50, min_positive_frac * residuals_oneside.size):
        q_tail = float(np.quantile(positive, 1.0 - alpha))
    else:
        # Too few violations to reliably estimate the tail; rely on q_scale / q_min.
        q_tail = 0.0

    iqr_b = float(np.quantile(signed_b, 0.75) - np.quantile(signed_b, 0.25))
    q_scale = lambda_scale * max(iqr_b, 0.0)

    q_hat = max(q_tail, q_scale, q_min)

    stats = {
        # Final threshold and its three components (paper diagnostic)
        "btccq/q_hat": q_hat,
        "btccq/q_tail": q_tail,
        "btccq/q_scale": q_scale,
        "btccq/q_min": q_min,
        # Residual-side diagnostics
        "btccq/calib_e_mean": float(residuals_oneside.mean()),
        "btccq/calib_e_std": float(residuals_oneside.std()),
        "btccq/calib_zero_frac": float((residuals_oneside <= eps_res).mean()),
        "btccq/calib_positive_frac": positive_frac,
        "btccq/calib_positive_n": int(positive.size),
        "btccq/calib_positive_mean": (
            float(positive.mean()) if positive.size > 0 else 0.0
        ),
        "btccq/calib_positive_p50": (
            float(np.quantile(positive, 0.50)) if positive.size > 0 else 0.0
        ),
        "btccq/calib_positive_p90": (
            float(np.quantile(positive, 0.90)) if positive.size > 0 else 0.0
        ),
        # Signed-residual scale diagnostics
        "btccq/calib_b_iqr": iqr_b,
        "btccq/calib_b_p25": float(np.quantile(signed_b, 0.25)),
        "btccq/calib_b_p75": float(np.quantile(signed_b, 0.75)),
        "btccq/calib_n": int(residuals_oneside.size),
    }
    return q_hat, stats


def main(_):
    """
    house keeping
    """
    assert FLAGS.online_sampling_method in [
        "mixed",
        "append",
    ], "incorrect online sampling method"

    if FLAGS.use_redq:
        FLAGS.config.agent_kwargs = add_redq_config(FLAGS.config.agent_kwargs)

    min_steps_to_update = FLAGS.batch_size * (1 - FLAGS.offline_data_ratio)
    if FLAGS.agent in ("calql", "btccq"):
        # btccq inherits SACAgent (no MC returns needed), but the warmup
        # logic is identical: collect at least one full episode of online
        # data before the first update so replay_buffer.sample() doesn't
        # raise ValueError on a too-small buffer.
        min_steps_to_update = max(
            min_steps_to_update, gym.make(FLAGS.env)._max_episode_steps
        )

    """
    wandb and logging
    """
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": "wsrl" or FLAGS.project,
            "group": "wsrl" or FLAGS.group,
            "exp_descriptor": f"{FLAGS.exp_name}_{FLAGS.env}_{FLAGS.agent}_seed{FLAGS.seed}",
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=FLAGS.config.to_dict(),
        random_str_in_identifier=True,
        disable_online_logging=FLAGS.debug,
    )

    save_dir = os.path.join(
        FLAGS.save_dir,
        wandb_logger.config.project,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
    )

    """
    env
    """
    # do not clip adroit actions online following CalQL repo
    # https://github.com/nakamotoo/Cal-QL
    env_type = get_env_type(FLAGS.env)
    finetune_env = make_gym_env(
        env_name=FLAGS.env,
        reward_scale=FLAGS.reward_scale,
        reward_bias=FLAGS.reward_bias,
        scale_and_clip_action=env_type in ("antmaze", "kitchen", "locomotion"),
        action_clip_lim=FLAGS.clip_action,
        seed=FLAGS.seed,
    )
    eval_env = make_gym_env(
        env_name=FLAGS.env,
        scale_and_clip_action=env_type in ("antmaze", "kitchen", "locomotion"),
        action_clip_lim=FLAGS.clip_action,
        seed=FLAGS.seed + 1000,
    )

    """
    load dataset
    """
    if env_type == "adroit-binary":
        dataset = get_hand_dataset_with_mc_calculation(
            FLAGS.env,
            gamma=FLAGS.config.agent_kwargs.discount,
            reward_scale=FLAGS.reward_scale,
            reward_bias=FLAGS.reward_bias,
            clip_action=FLAGS.clip_action,
        )
    else:
        # btccq v2 inherits SACAgent (no CQL/MC), so it doesn't need mc_returns.
        # We therefore only ask for the MC dataset for actual calql runs.
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

    """
    replay buffer
    """
    # btccq is now SAC-based and does not need MC returns.
    _needs_mc = FLAGS.agent == "calql"
    replay_buffer_type = ReplayBufferMC if _needs_mc else ReplayBuffer
    replay_buffer = replay_buffer_type(
        finetune_env.observation_space,
        finetune_env.action_space,
        capacity=FLAGS.replay_buffer_capacity,
        seed=FLAGS.seed,
        discount=FLAGS.config.agent_kwargs.discount if _needs_mc else None,
    )

    """
    Fixed diagnostic batch (sampled once from offline data).
    Used at every eval to compute Q-collapse statistics on a stable distribution.
    """
    _diag_rng = np.random.default_rng(FLAGS.seed)
    _diag_n = min(2048, dataset["observations"].shape[0])
    _diag_idx = _diag_rng.choice(
        dataset["observations"].shape[0], size=_diag_n, replace=False
    )
    fixed_diag_batch = {
        "observations": jax.numpy.asarray(dataset["observations"][_diag_idx]),
        "actions": jax.numpy.asarray(dataset["actions"][_diag_idx]),
    }

    """
    Initialize agent.

    For --agent btccq we build a SACAgent first to load the pretrained
    checkpoint into (BTCCQAgent inherits SACAgent and has an extra
    offline_params pytree field that has no value yet); we wrap into
    BTCCQAgent below after calibration. Critic + actor architecture in the
    v2 config is chosen to match the CalQL pretrain checkpoint exactly
    (2-Q ensemble, no LayerNorm), so Orbax restore is clean and the only
    parameters that get dropped are the CQL-specific lagrange multiplier
    (which SACAgent doesn't have and doesn't need).
    """
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, construct_rng = jax.random.split(rng)
    example_batch = subsample_batch(dataset, FLAGS.batch_size)
    _create_agent_name = "sac" if FLAGS.agent == "btccq" else FLAGS.agent
    agent = agents[_create_agent_name].create(
        rng=construct_rng,
        observations=example_batch["observations"],
        actions=example_batch["actions"],
        encoder_def=None,
        **FLAGS.config.agent_kwargs,
    )

    if FLAGS.resume_path != "":
        assert os.path.exists(FLAGS.resume_path), "resume path does not exist"
        agent = checkpoints.restore_checkpoint(FLAGS.resume_path, target=agent)

    # BT-CCQ wrap is deferred to the offline -> online transition (see
    # the training loop below). Until then the agent is a vanilla SAC
    # agent (whether built from scratch or restored from a checkpoint),
    # which means the offline phase trains as pure SAC -- no gate yet.
    btccq_pending_wrap = FLAGS.agent == "btccq"

    """
    eval function
    """

    def evaluate_and_log_results(
        eval_env,
        policy_fn,
        eval_func,
        step_number,
        wandb_logger,
        n_eval_trajs=FLAGS.n_eval_trajs,
    ):
        stats, trajs = eval_func(
            policy_fn,
            eval_env,
            n_eval_trajs,
        )

        eval_info = {
            "average_return": np.mean([np.sum(t["rewards"]) for t in trajs]),
            "average_traj_length": np.mean([len(t["rewards"]) for t in trajs]),
        }
        if env_type == "adroit-binary":
            # adroit
            eval_info["success_rate"] = np.mean(
                [any(d["goal_achieved"] for d in t["infos"]) for t in trajs]
            )
        elif env_type == "kitchen":
            # kitchen
            eval_info["num_stages_solved"] = np.mean([t["rewards"][-1] for t in trajs])
            eval_info["success_rate"] = np.mean([t["rewards"][-1] for t in trajs]) / 4
        else:
            # d4rl antmaze, locomotion
            eval_info["success_rate"] = eval_info[
                "average_normalized_return"
            ] = np.mean(
                [eval_env.get_normalized_score(np.sum(t["rewards"])) for t in trajs]
            )

        wandb_logger.log({"evaluation": eval_info}, step=step_number)

    """
    training loop
    """
    timer = Timer()
    step = int(agent.state.step)  # 0 for new agents, or load from pre-trained

    # If we restored from a checkpoint (step > 0) but the user passed
    # --num_offline_steps 0 (skip pretrain), the original
    # range(step, num_offline_steps + num_online_steps) becomes empty
    # because num_offline_steps + num_online_steps < step. Lift the upper
    # bound so we always run num_online_steps of online updates.
    _final_step = max(
        FLAGS.num_offline_steps + FLAGS.num_online_steps,
        step + FLAGS.num_online_steps,
    )

    is_online_stage = False
    observation, info = finetune_env.reset()
    done = False  # env done signal

    for _ in tqdm.tqdm(range(step, _final_step)):
        """
        Switch from offline to online
        """
        if not is_online_stage and step >= FLAGS.num_offline_steps:
            logging.info("Switching to online training")
            is_online_stage = True

            # upload offline data to online buffer
            if FLAGS.online_sampling_method == "append":
                offline_dataset_size = dataset["actions"].shape[0]
                dataset_items = dataset.items()
                for j in range(offline_dataset_size):
                    transition = {k: v[j] for k, v in dataset_items}
                    replay_buffer.insert(transition)

            # option for CQL / CalQL to change the online alpha and CQL regularizer.
            # btccq is now SAC-based -- no CQL knobs to update.
            if FLAGS.agent in ("cql", "calql"):
                online_agent_configs = {
                    "cql_alpha": FLAGS.config.agent_kwargs.get(
                        "online_cql_alpha", None
                    ),
                    "use_cql_loss": FLAGS.online_use_cql_loss,
                }
                agent.update_config(online_agent_configs)

            # CFS-D: at the offline -> online boundary, score the frozen
            # REDQ heads and optionally restrict online REDQ target-head
            # sampling to the selected low-footprint / low-influence pool.
            if FLAGS.use_cfs:
                logging.info("CFS-D: running transition-time calibration...")
                critic_subsample_size = FLAGS.config.agent_kwargs.get(
                    "critic_subsample_size", 2
                )
                agent, cfs_info = maybe_apply_cfs_calibration(
                    agent=agent,
                    dataset=dataset,
                    gamma=FLAGS.config.agent_kwargs.discount,
                    use_cfs=True,
                    mode=FLAGS.cfs_mode,
                    top_k=FLAGS.cfs_top_k,
                    num_samples=FLAGS.cfs_calib_n,
                    batch_size=FLAGS.cfs_calib_batch_size,
                    e_weight=FLAGS.cfs_e_weight,
                    critic_subsample_size=critic_subsample_size,
                    dominance_samples=FLAGS.cfs_dominance_samples,
                    min_cv=FLAGS.cfs_min_cv,
                    require_heterogeneity=FLAGS.cfs_require_heterogeneity,
                    output_path=FLAGS.cfs_stats_output,
                    seed=FLAGS.seed,
                )
                logging.info("CFS-D calibration: %s", cfs_info)
                wandb_logger.log({"cfs_calibration": cfs_info}, step=step)

            # BT-CCQ: at the offline -> online boundary, freeze the
            # current SAC params as the offline reference, calibrate
            # q_hat on the d4rl dataset, then wrap the live SACAgent
            # into a BTCCQAgent so the gate kicks in for online updates.
            if btccq_pending_wrap:
                logging.info("BT-CCQ: snapshotting offline params at transition")
                offline_params = jax.device_get(agent.state.params)

                logging.info("BT-CCQ: running offline calibration...")
                q_hat, calib_stats = calibrate_qhat(
                    agent=agent,
                    dataset=dataset,
                    gamma=FLAGS.config.agent_kwargs.discount,
                    alpha=FLAGS.btccq_alpha,
                    calib_ratio=FLAGS.btccq_calib_ratio,
                )
                logging.info("BT-CCQ calibration: %s", calib_stats)
                wandb_logger.log({"btccq_calibration": calib_stats}, step=step)

                agent = BTCCQAgent.create_from_sac(
                    sac_agent=agent,
                    offline_params=offline_params,
                    q_hat=q_hat,
                    w_out=FLAGS.btccq_w_out,
                    eps=FLAGS.btccq_eps,
                )
                logging.info(
                    "BT-CCQ: agent wrapped, q_hat=%.4f, w_out=%.2f",
                    q_hat,
                    FLAGS.btccq_w_out,
                )
                btccq_pending_wrap = False

        timer.tick("total")

        """
        Env Step
        """
        with timer.context("env step"):
            if is_online_stage:
                rng, action_rng = jax.random.split(rng)
                action = agent.sample_actions(observation, seed=action_rng)
                next_observation, reward, done, truncated, info = finetune_env.step(
                    action
                )

                transition = dict(
                    observations=observation,
                    next_observations=next_observation,
                    actions=action,
                    rewards=reward,
                    masks=1.0 - done,
                    dones=1.0 if (done or truncated) else 0,
                )
                replay_buffer.insert(transition)

                observation = next_observation
                if done or truncated:
                    observation, info = finetune_env.reset()
                    done = False

        """
        Updates
        """
        with timer.context("update"):
            # offline updates
            if not is_online_stage:
                batch = subsample_batch(dataset, FLAGS.batch_size)
                agent, update_info = agent.update(
                    batch,
                )

            # online updates
            else:
                if step - FLAGS.num_offline_steps <= max(
                    FLAGS.warmup_steps, min_steps_to_update
                ):
                    # no updates during warmup
                    pass
                else:
                    # do online updates, gather batch
                    batch = None  # sentinel: skip update if not enough data
                    if FLAGS.online_sampling_method == "mixed":
                        batch_size_offline = int(
                            FLAGS.batch_size * FLAGS.offline_data_ratio
                        )
                        batch_size_online = FLAGS.batch_size - batch_size_offline
                        # Defensive: skip update if online buffer doesn't yet
                        # have enough samples (avoids ValueError in
                        # replay_buffer.sample when buffer < batch).
                        if (
                            batch_size_online == 0
                            or len(replay_buffer) >= batch_size_online
                        ):
                            online_batch = replay_buffer.sample(batch_size_online)
                            offline_batch = subsample_batch(dataset, batch_size_offline)
                            batch = concatenate_batches([online_batch, offline_batch])
                    elif FLAGS.online_sampling_method == "append":
                        if len(replay_buffer) >= FLAGS.batch_size:
                            batch = replay_buffer.sample(FLAGS.batch_size)
                    else:
                        raise RuntimeError("Incorrect online sampling method")

                    # update (only if we got a batch)
                    if batch is not None:
                        if FLAGS.utd > 1:
                            agent, update_info = agent.update_high_utd(
                                batch,
                                utd_ratio=FLAGS.utd,
                            )
                        else:
                            agent, update_info = agent.update(
                                batch,
                            )

        """
        Advance Step
        """
        step += 1

        """
        Evals
        """
        eval_steps = (
            FLAGS.num_offline_steps,  # finish offline training
            FLAGS.num_offline_steps + 1,  # start of online training
            FLAGS.num_offline_steps + FLAGS.num_online_steps,  # end of online training
        )
        if step % FLAGS.eval_interval == 0 or step in eval_steps:
            logging.info("Evaluating...")
            with timer.context("evaluation"):
                policy_fn = partial(
                    agent.sample_actions, argmax=FLAGS.deterministic_eval
                )
                eval_func = partial(
                    evaluate_with_trajectories, clip_action=FLAGS.clip_action
                )

                evaluate_and_log_results(
                    eval_env=eval_env,
                    policy_fn=policy_fn,
                    eval_func=eval_func,
                    step_number=step,
                    wandb_logger=wandb_logger,
                )

                # Q-collapse diagnostic on fixed offline batch
                q_diag_stats = compute_fixed_q_stats(agent, fixed_diag_batch)
                wandb_logger.log({"q_diag": q_diag_stats}, step=step)

        """
        Save Checkpoint
        """
        if step % FLAGS.save_interval == 0 or step == FLAGS.num_offline_steps:
            logging.info("Saving checkpoint...")
            checkpoint_path = checkpoints.save_checkpoint(
                save_dir, agent, step=step, keep=30
            )
            logging.info("Saved checkpoint to %s", checkpoint_path)

        timer.tock("total")

        """
        Logging
        """
        if step % FLAGS.log_interval == 0:
            # check if update_info is available (False during warmup)
            if "update_info" in locals():
                update_info = jax.device_get(update_info)
                wandb_logger.log({"training": update_info}, step=step)

            wandb_logger.log({"timer": timer.get_average_times()}, step=step)


if __name__ == "__main__":
    app.run(main)
