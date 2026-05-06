"""
BT-CCQ agent.

Extends CalQLAgent with a gate-weighted critic TD loss.
Everything else — actor, temperature, target update, high-UTD scan,
replay, warmup, evaluation — is untouched.

The only algorithmic change:
    critic_loss = mean(gate * (Q - y_live)^2)

where
    delta   = max(0, z_off - y_live)
    gate    = 1                         if delta <= q_hat
            = max(w_out, q_hat/delta)   otherwise

and
    z_off = r + gamma * masks * V_off(s')
    V_off(s') = min_j Q_off^j(s', pi_off(s'))

Frozen offline params are stored as pytree fields so JAX can trace through
them inside JIT.  q_hat, w_out, eps are stored in config (nonpytree_field)
as plain Python scalars — fine because they never change.
"""

from functools import partial
from typing import Optional, Tuple

import chex
import flax
import jax
import jax.numpy as jnp
from overrides import overrides

from wsrl.agents.calql import CalQLAgent
from wsrl.common.typing import Batch, Params, PRNGKey


class BTCCQAgent(CalQLAgent):
    """
    CalQL + BT-CCQ gate on the critic TD loss.

    Extra fields (all pytree nodes so JAX can differentiate / JIT through):
        offline_params: frozen copy of the CalQL-pretrained params

    Extra config keys (nonpytree scalars, set at construction time):
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
        Gate-weighted TD loss.  Calls super() to get the standard CalQL
        critic loss, then replaces the MSE term with gate * MSE.
        """
        batch_size = batch["rewards"].shape[0]
        rng, next_action_sample_key = jax.random.split(rng)

        # ------------------------------------------------------------------
        # 1. Live TD target  y_live  (same as CalQL / SAC)
        # ------------------------------------------------------------------
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch, next_action_sample_key
        )

        target_next_qs = self.forward_target_critic(
            batch["next_observations"],
            next_actions,
            rng=rng,
        )  # (ensemble_size, batch_size)

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

        # pi_off(s')
        a_next_off = self.state.apply_fn(
            {"params": self.offline_params},
            next_obs,
            name="actor",
            rngs={"dropout": rng},
            train=False,
        ).mode()  # deterministic; distrax Distribution.mode()

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
            batch["rewards"]
            + self.config["discount"] * batch["masks"] * v_off_next
        )  # (batch_size,)

        # ------------------------------------------------------------------
        # 3. BT-CCQ gate  (pure JAX, JIT-safe)
        # ------------------------------------------------------------------
        q_hat = self.config["btccq_q_hat"]
        w_out = self.config.get("btccq_w_out", 0.2)
        eps   = self.config.get("btccq_eps",   1e-6)

        delta      = jnp.maximum(0.0, z_off - y_live)           # (batch_size,)
        soft_w     = jnp.clip(q_hat / (delta + eps), w_out, 1.0)
        gate       = jnp.where(delta > q_hat, soft_w, 1.0)      # (batch_size,)

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
        # 5. Gate-weighted MSE loss
        # ------------------------------------------------------------------
        target_qs = y_live[None].repeat(self.config["critic_ensemble_size"], axis=0)
        # gate broadcast: (1, batch_size) → (ensemble_size, batch_size)
        td_sq     = (predicted_qs - target_qs) ** 2              # (E, N)
        critic_loss = jnp.mean(gate[None] * td_sq)

        # ------------------------------------------------------------------
        # 6. Info dict
        # ------------------------------------------------------------------
        info = {
            "critic_loss":       critic_loss,
            "predicted_qs":      jnp.mean(predicted_qs),
            "target_qs":         jnp.mean(y_live),
            "btccq/q_hat":       q_hat,
            "btccq/z_off_mean":  jnp.mean(z_off),
            "btccq/y_live_mean": jnp.mean(y_live),
            "btccq/delta_mean":  jnp.mean(delta),
            "btccq/delta_max":   jnp.max(delta),
            "btccq/gate_mean":   jnp.mean(gate),
            "btccq/gate_min":    jnp.min(gate),
            "btccq/gate_frac":   jnp.mean(delta > q_hat),
        }

        return critic_loss, info

    @classmethod
    def create_from_calql(
        cls,
        calql_agent: CalQLAgent,
        offline_params: Params,
        q_hat: float,
        w_out: float = 0.2,
        eps: float = 1e-6,
    ) -> "BTCCQAgent":
        """
        Construct a BTCCQAgent from an already-created (and optionally
        checkpoint-restored) CalQLAgent.

        Args:
            calql_agent:    the restored CalQL agent (used as the live agent)
            offline_params: frozen copy of the offline checkpoint params
            q_hat:          calibrated BT-CCQ threshold
            w_out:          minimum gate weight
            eps:            numerical stability constant
        """
        new_config = dict(calql_agent.config)
        new_config["btccq_q_hat"] = float(q_hat)
        new_config["btccq_w_out"] = float(w_out)
        new_config["btccq_eps"]   = float(eps)

        return cls(
            state=calql_agent.state,
            config=new_config,
            offline_params=offline_params,
        )
