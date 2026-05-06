from ml_collections import ConfigDict

from experiments.configs.cql_config import get_config as get_cql_config
from experiments.configs.iql_config import get_config as get_iql_config
from experiments.configs.sac_config import get_config as get_sac_config
from experiments.configs.wsrl_config import get_config as get_wsrl_config


def get_btccq_config(updates=None):
    """CalQL-based config for BT-CCQ: CQL foundation + WSRL-style network arch."""
    return get_cql_config(
        updates=dict(
            use_calql=True,
            calql_bound_random_actions=False,
            cql_autotune_alpha=True,
            cql_target_action_gap=0.8,
            online_cql_alpha=0.0,  # drop CQL penalty online; gate handles OOD
            **(updates or {}),
        )
    )


def get_btccq_v2_config(updates=None):
    """
    Pure SAC config for BT-CCQ v2 (WSRL-aligned).

    Inherits sac_config defaults so there are no CQL keys at all. The
    v2 BTCCQAgent inherits SACAgent and applies only the gate, so this
    config must NOT contain CQL keys (cql_n_actions, cql_alpha, etc) —
    they would just be ignored, but keeping them clean avoids confusion.
    """
    return get_sac_config(updates=updates)


def get_config(config_string):

    possible_structures = {

        ########################################################
        #                    antmaze configs                   #
        ########################################################

        "antmaze_cql": ConfigDict(
            dict(
                agent_kwargs=get_cql_config(
                    updates=dict(
                        policy_kwargs=dict(
                            tanh_squash_distribution=True,
                            std_parameterization="uniform",
                        ),
                        critic_network_kwargs={
                            "hidden_dims": [256, 256, 256, 256],
                            "activations": "relu",
                            "kernel_scale_final": 1e-2,
                        },
                        policy_network_kwargs={
                            "hidden_dims": [256, 256],
                            "activations": "relu",
                            "kernel_scale_final": 1e-2,
                        },
                        cql_autotune_alpha=True,
                        cql_target_action_gap=0.8,
                    )
                ).to_dict(),
            )
        ),

        "antmaze_iql":ConfigDict(
            dict(
                agent_kwargs=get_iql_config(
                    updates=dict(
                        expectile=0.9,
                        temperature=10.0,
                    )
                ).to_dict(),
            )
        ),

        "antmaze_btccq": ConfigDict(
            dict(
                # NOTE: arch must match `antmaze_cql` exactly because BT-CCQ
                # restores from the antmaze_cql CalQL checkpoint. WSRL adds
                # use_layer_norm=True but our CalQL pretrain didn't, so we
                # also keep it off here to avoid uninitialised LayerNorm
                # parameters at restore time.
                agent_kwargs=get_btccq_config(
                    updates=dict(
                        policy_kwargs=dict(
                            tanh_squash_distribution=True,
                            std_parameterization="uniform",
                        ),
                        critic_network_kwargs={
                            "hidden_dims": [256, 256, 256, 256],
                            "activations": "relu",
                            "kernel_scale_final": 1e-2,
                        },
                        policy_network_kwargs={
                            "hidden_dims": [256, 256],
                            "activations": "relu",
                            "kernel_scale_final": 1e-2,
                        },
                        max_target_backup=True,
                    )
                ).to_dict(),
            )
        ),

        "antmaze_btccq_v2": ConfigDict(
            dict(
                # WSRL-aligned BT-CCQ: pure SAC online + BT gate (no CQL anywhere).
                # Architecture matches the antmaze_cql pretrain checkpoint:
                #   - 2-head double-Q critic (sac_config default)
                #   - no LayerNorm (sac_config default; CalQL pretrain didn't use it)
                #   - same critic/policy hidden dims
                # The only "WSRL-style" deviation from sac_config is
                # backup_entropy=True (standard in modern SAC).
                agent_kwargs=get_btccq_v2_config(
                    updates=dict(
                        backup_entropy=True,
                        policy_kwargs=dict(
                            tanh_squash_distribution=True,
                            std_parameterization="uniform",
                        ),
                        critic_network_kwargs={
                            "hidden_dims": [256, 256, 256, 256],
                            "activations": "relu",
                            "kernel_scale_final": 1e-2,
                            "use_layer_norm": False,
                        },
                        policy_network_kwargs={
                            "hidden_dims": [256, 256],
                            "activations": "relu",
                            "kernel_scale_final": 1e-2,
                            "use_layer_norm": False,
                        },
                    )
                ).to_dict(),
            )
        ),

        "antmaze_wsrl": ConfigDict(
            dict(
                # NOTE: arch must match `antmaze_cql` exactly because reduced/full
                # WSRL restore from the antmaze_cql CalQL pretrain checkpoint.
                # Two overrides from upstream wsrl_config defaults:
                #   - use_layer_norm=False (CalQL pretrain has no LayerNorm)
                #   - critic_ensemble_size=2 / critic_subsample_size=None
                #     (CalQL pretrain uses 2-head double-Q, not 10-head REDQ).
                #   Without these, Orbax restore silently leaves uninitialised
                #   parameters and a chex shape assert fails on the first
                #   gradient step.
                agent_kwargs=get_wsrl_config(
                    updates=dict(
                        critic_ensemble_size=2,
                        critic_subsample_size=None,
                        policy_kwargs=dict(
                            tanh_squash_distribution=True,
                            std_parameterization="uniform",
                        ),
                        critic_network_kwargs={
                            "hidden_dims": [256, 256, 256, 256],
                            "activations": "relu",
                            "kernel_scale_final": 1e-2,
                            "use_layer_norm": False,
                        },
                        policy_network_kwargs={
                            "hidden_dims": [256, 256],
                            "activations": "relu",
                            "kernel_scale_final": 1e-2,
                            "use_layer_norm": False,
                        },
                        max_target_backup=True,
                    )
                ).to_dict(),
            )
        ),

        ########################################################
        #                    adroit configs                    #
        ########################################################

        "adroit_cql": ConfigDict(
            dict(
                agent_kwargs=get_cql_config(
                    updates=dict(
                        policy_kwargs=dict(
                            tanh_squash_distribution=True,
                            std_parameterization="exp",
                        ),
                        critic_network_kwargs={
                            "hidden_dims": [512, 512, 512],
                            "kernel_scale_final": 1e-2,
                            "activations": "relu",
                        },
                        policy_network_kwargs={
                            "hidden_dims": [512, 512],
                            "kernel_scale_final": 1e-2,
                            "activations": "relu",
                        },
                        online_cql_alpha=1.0,
                        cql_alpha=1.0,
                    )
                ).to_dict(),
            )
        ),

        "adroit_iql":ConfigDict(
            dict(
                agent_kwargs=get_iql_config(
                    updates=dict(
                        policy_network_kwargs=dict(
                            hidden_dims=(256, 256),
                            kernel_init_type="var_scaling",
                            kernel_scale_final=1e-2,
                            dropout_rate=0.1,
                        ),
                        expectile=0.7,
                        temperature=0.5,
                    ),
                ).to_dict(),
            )
        ),

        "adroit_wsrl": ConfigDict(
            dict(
                agent_kwargs=get_wsrl_config(
                    updates=dict(
                        policy_kwargs=dict(
                            tanh_squash_distribution=True,
                            std_parameterization="exp",
                        ),
                        critic_network_kwargs={
                            "hidden_dims": [512, 512, 512],
                            "kernel_scale_final": 1e-2,
                            "activations": "relu",
                            "use_layer_norm": True,
                        },
                        policy_network_kwargs={
                            "hidden_dims": [512, 512],
                            "kernel_scale_final": 1e-2,
                            "activations": "relu",
                            "use_layer_norm": True,
                        },
                    )
                ).to_dict(),
            )
        ),

        ########################################################
        #                    kitchen configs                   #
        ########################################################

        "kitchen_cql": ConfigDict(
            dict(
                agent_kwargs=get_cql_config(
                    updates=dict(
                        policy_kwargs=dict(
                            tanh_squash_distribution=True,
                            std_parameterization="exp",
                        ),
                        critic_network_kwargs={
                            "hidden_dims": [512, 512, 512],
                            "activations": "relu",
                        },
                        policy_network_kwargs={
                            "hidden_dims": [512, 512, 512],
                            "activations": "relu",
                        },
                        online_cql_alpha=5.0,
                        cql_alpha=5.0,
                        cql_importance_sample=False,
                    )
                ).to_dict(),
            )
        ),

        "kitchen_iql":ConfigDict(
            dict(
                agent_kwargs=get_iql_config(
                    updates=dict(
                        policy_network_kwargs=dict(
                            hidden_dims=(256, 256),
                            activations="relu",
                            dropout_rate=0.1,
                        ),
                        critic_network_kwargs=dict(
                            hidden_dims=(256, 256),
                            activations="relu",
                        ),
                        expectile=0.7,
                        temperature=0.5,
                    )
                ).to_dict(),
            )
        ),

        "kitchen_wsrl": ConfigDict(
            dict(
                agent_kwargs=get_wsrl_config(
                    updates=dict(
                        policy_kwargs=dict(
                            tanh_squash_distribution=True,
                            std_parameterization="exp",
                        ),
                        critic_network_kwargs={
                            "hidden_dims": [512, 512, 512],
                            "activations": "relu",
                            "use_layer_norm": True,
                        },
                        policy_network_kwargs={
                            "hidden_dims": [512, 512, 512],
                            "activations": "relu",
                            "use_layer_norm": True,
                        },
                    )
                ).to_dict(),
            )
        ),

        ########################################################
        #                  locomotion configs                  #
        ########################################################

        "locomotion_cql": ConfigDict(
            dict(
                agent_kwargs=get_cql_config(
                    updates=dict(
                        critic_network_kwargs={
                            "hidden_dims": [256, 256],
                            "activations": "relu",
                            "kernel_scale_final": 1e-2,
                        },
                        policy_network_kwargs={
                            "hidden_dims": [256, 256],
                            "activations": "relu",
                            "kernel_scale_final": 1e-2,
                        },
                        online_cql_alpha=5.0,
                        cql_alpha=5.0,
                    )
                ).to_dict(),
            )
        ),

        "locomotion_iql":ConfigDict(
            dict(
                agent_kwargs=get_iql_config(
                    updates=dict(
                        expectile=0.7,
                        temperature=3.0,
                    )
                ).to_dict(),
            )
        ),

        "locomotion_wsrl": ConfigDict(
            dict(
                agent_kwargs=get_wsrl_config(
                    updates=dict(
                        critic_network_kwargs={
                            "hidden_dims": [256, 256],
                            "activations": "relu",
                            "kernel_scale_final": 1e-2,
                            "use_layer_norm": True,
                        },
                        policy_network_kwargs={
                            "hidden_dims": [256, 256],
                            "activations": "relu",
                            "kernel_scale_final": 1e-2,
                            "use_layer_norm": True,
                        },
                    )
                ).to_dict(),
            )
        ),
    }

    return possible_structures[config_string]
