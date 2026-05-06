"""
BT-CCQ agent (WSRL-aligned).

Inherits SACAgent (NOT CalQLAgent) so the online fine-tuning objective is
strictly:
    critic_loss = mean( gate(s,a) * (Q(s,a) - y_live)^2 )
                  + (no CQL penalty)

where
    delta(s,a) = max(0, z_off(s,a) - y_live(s,a))
    gate       = 1                          if delta <= q_hat
               = max(w_out, q_hat/delta)    otherwise

and
    z_off = r + gamma * masks * V_off(s')
    V_off(s') = min_j Q_off^j(s', pi_off(s'))

Frozen offline params are stored as a pytree field so JAX can trace through
them inside JIT.  q_hat, w_out, eps are stored in config (nonpytree_field)
as plain Python scalars — fine because they never change.

Key design choice (relative to earlier CalQL-based BTCCQAgent):
- Inheriting SACAgent removes the CQL pessimism path entirely. The gate is
  the *only* mechanism modulating the critic objective, which matches the
  WSRL "online SAC fine-tune from offline init" setting in the original
  paper. Earlier prototype inherited CalQLAgent which left
  cql_alpha_lagrange / cql_max_target_backup / CQL action sampling active
  during online training and confounded the BT-CCQ ablation.
"""

from typing import Optional

import chex
import jax
import jax.numpy as jnp
from overrides import overrides

from wsrl.agents.sac import SACAgent
from wsrl.common.typing import Batch, Params, PRNGKey


class BTCCQAgent(SACAgent):
    """
    SAC + BT-CCQ gate on the critic TD loss.

    Extra pytree field:
        offline_params: frozen copy of the pretrained CalQL params (or
                        any pretrained Q/policy params with the same arch)

    Extra nonpytree config keys:
        btccq_q_hat   float   calibrated threshold
        btccq_w_out   float   minimum gate weight (default 0.2)
        btccq_eps     float   numerical stability (default 1e-6)
    """

    # Frozen offline params stored as a pytree field.
    # JAX treats them as regular arrays — they will be traced but never updated.
    offline_params: Params

    @overrides
    def critic_loss_fn(self, batch: Batch, params: Params, rng: PRNGKey):
        """
        Pure SAC TD loss × gate. No CQL penalty, no CQL max-target-backup,
        no CQL n-action sampling.
        """
        batch_size = batch["rewards"].shape[0]
        rng, next_action_sample_key = jax.random.split(rng)

        # ------------------------------------------------------------------
        # 1. Live TD target  y_live  (standard SAC, no CQL specials)
        # ------------------------------------------------------------------
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch, next_action_sample_key
        )

        target_next_qs = self.forward_target_critic(
            batch["next_observations"],
            next_actions,
            rng=rng,
        )  # (ensemble_size, batch_size)

        # REDQ-style subsample if requested by config
        if self.config["critic_subsample_size"] is not None:
            rng, subsample_key = jax.random.split(rng)
            subsample_idcs = jax.random.randint(
                subsample_key,
                (self.config["critic_subsample_size"],),
                0,
                self.config["critic_ensemble_size"],
            )
            target_next_qs = target_next_qs[subsample_idcs]

        target_next_min_q = target_next_qs.min(axis=0)  # (batch_size,)

        # SAC entropy backup (NOT CQL backup)
        target_next_min_q = self._process_target_next_qs(
            target_next_min_q, next_actions_log_probs
        )

        y_live = (
            batch["rewards"]
            + self.config["discount"] * batch["masks"] * target_next_min_q
        )  # (batch_size,)
        chex.assert_shape(y_live, (batch_size,))

        # ------------------------------------------------------------------
        # 2. Offline Bellman reference  z_off  (frozen params, no grad)
        # ------------------------------------------------------------------
        next_obs = batch["next_observations"]

        # pi_off(s')  — deterministic mode of the frozen pretrained policy
        a_next_off = self.state.apply_fn(
            {"params": self.offline_params},
            next_obs,
            name="actor",
            rngs={"dropout": rng},
            train=False,
        ).mode()

        # Q_off(s', pi_off(s'))  →  (ensemble_size, batch_size)
        q_next_off = self.state.apply_fn(
            {"params": self.offline_params},
            next_obs,
            a_next_off,
            name="critic",
            rngs={"dropout": rng},
            train=False,
        )
        v_off_next = q_next_off.min(axis=0)  # (batch_size,)

        z_off = (
            batch["rewards"] + self.config["discount"] * batch["masks"] * v_off_next
        )  # (batch_size,)

        # ------------------------------------------------------------------
        # 3. BT-CCQ gate  (pure JAX, JIT-safe)
        # ------------------------------------------------------------------
        q_hat = self.config["btccq_q_hat"]
        w_out = self.config.get("btccq_w_out", 0.2)
        eps = self.config.get("btccq_eps", 1e-6)

        delta = jnp.maximum(0.0, z_off - y_live)  # (batch_size,)
        soft_w = jnp.clip(q_hat / (delta + eps), w_out, 1.0)
        gate = jnp.where(delta > q_hat, soft_w, 1.0)  # (batch_size,)

        # ------------------------------------------------------------------
        # 4. Predicted Q values
        # ------------------------------------------------------------------
        predicted_qs = self.forward_critic(
            batch["observations"],
            batch["actions"],
            rng=rng,
            grad_params=params,
        )  # (ensemble_size, batch_size)
        chex.assert_shape(
            predicted_qs, (self.config["critic_ensemble_size"], batch_size)
        )

        # ------------------------------------------------------------------
        # 5. Gate-weighted MSE loss (the only critic objective)
        # ------------------------------------------------------------------
        target_qs = y_live[None].repeat(self.config["critic_ensemble_size"], axis=0)
        td_sq = (predicted_qs - target_qs) ** 2  # (E, N)
        critic_loss = jnp.mean(gate[None] * td_sq)

        info = {
            "critic_loss": critic_loss,
            "predicted_qs": jnp.mean(predicted_qs),
            "target_qs": jnp.mean(y_live),
            "btccq/q_hat": q_hat,
            "btccq/z_off_mean": jnp.mean(z_off),
            "btccq/y_live_mean": jnp.mean(y_live),
            "btccq/delta_mean": jnp.mean(delta),
            "btccq/delta_max": jnp.max(delta),
            "btccq/gate_mean": jnp.mean(gate),
            "btccq/gate_min": jnp.min(gate),
            "btccq/gate_frac": jnp.mean(delta > q_hat),
        }

        return critic_loss, info

    @classmethod
    def create_from_sac(
        cls,
        sac_agent: SACAgent,
        offline_params: Params,
        q_hat: float,
        w_out: float = 0.2,
        eps: float = 1e-6,
    ) -> "BTCCQAgent":
        """
        Construct a BTCCQAgent from an already-created (and optionally
        checkpoint-restored) SACAgent.

        Args:
            sac_agent:      the restored SAC agent (used as the live agent)
            offline_params: frozen copy of the offline checkpoint params
            q_hat:          calibrated BT-CCQ threshold
            w_out:          minimum gate weight
            eps:            numerical stability constant
        """
        # SACAgent stores config as a plain dict; CalQLAgent stores it as
        # FrozenDict. Plain dict.copy() takes no args, FrozenDict.copy(updates)
        # does. Handle both.
        updates = {
            "btccq_q_hat": float(q_hat),
            "btccq_w_out": float(w_out),
            "btccq_eps": float(eps),
        }
        old_config = sac_agent.config
        if isinstance(old_config, dict) and not hasattr(old_config, "freeze"):
            # plain dict (SACAgent path) -- shallow copy and update
            new_config = dict(old_config)
            new_config.update(updates)
        else:
            # FrozenDict / ConfigDict path
            new_config = old_config.copy(updates)

        return cls(
            state=sac_agent.state,
            config=new_config,
            offline_params=offline_params,
        )

    # Backwards-compat alias for older finetune.py call sites
    @classmethod
    def create_from_calql(cls, calql_agent, offline_params, q_hat, w_out=0.2, eps=1e-6):
        """Alias kept so finetune.py doesn't need to change."""
        return cls.create_from_sac(
            sac_agent=calql_agent,
            offline_params=offline_params,
            q_hat=q_hat,
            w_out=w_out,
            eps=eps,
        )
