"""
Hyperparameter sweep for JAX-based Box2D environments (LunarLander, BipedalWalker).
Uses JaxApproximateLearner + JaxSolver with warm-starting.

Usage:
    python -m tuning.sweep_box2d_jax tuning/configs/box2d/lunarlander_jax.yaml --agent-type expectation
    python -m tuning.sweep_box2d_jax tuning/configs/box2d/lunarlander_jax.yaml --agent-type variance
    python -m tuning.sweep_box2d_jax tuning/configs/box2d/lunarlander_jax.yaml --agent-type mmd
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.20"

import argparse
import torch
import wandb
import agents.demonstrator as demonstrator
from environments.box2d_jax_environment import Box2DJaxEnvironment
from environments.box2d_environment import Box2DEnvironment
from solvers.MDP_solver_approximation import MDPSolverApproximationExpectation
from tuning.common import (
    load_config,
    prepare_sweep_config,
    build_learner_config,
    log_memory,
    train_and_evaluate_jax,
    train_and_evaluate_mmd,
)

_yaml_config = None
_agent_type = None


def create_jax_environment(env_cfg: dict) -> Box2DJaxEnvironment:
    return Box2DJaxEnvironment(
        {
            "env_id": env_cfg["env_id"],
            "gamma": env_cfg.get("gamma", 0.99),
            "T": env_cfg["T"],
            "continuous": env_cfg.get("continuous", False),
        }
    )


def create_demo_environment(env_cfg: dict) -> Box2DEnvironment:
    """Create SB3-based environment for the demonstrator."""
    return Box2DEnvironment(
        {
            "env_id": env_cfg["env_id"],
            "gamma": env_cfg.get("gamma", 0.99),
            "T": env_cfg["T"],
            "continuous": env_cfg.get("continuous", False),
            "enable_wind": env_cfg.get("enable_wind", False),
        }
    )


def train():
    with wandb.init() as run:
        sweep_cfg = run.config
        env_cfg = _yaml_config["environment"]
        demo_cfg = _yaml_config["demonstrator"]

        # JAX environment for the learner
        env = create_jax_environment(env_cfg)
        # SB3 environment for the demonstrator
        demo_env = create_demo_environment(env_cfg)
        log_memory("env_config_creation")

        continuous = env_cfg.get("continuous", False)
        training_algorithm = getattr(
            sweep_cfg,
            "training_algorithm",
            "sac" if continuous else "dqn",
        )
        experiment_name = env_cfg["env_id"] + (
            "_continuous_jax" if continuous else "_discrete_jax"
        )

        # Training config for JAX DQN/SAC
        training_config = {
            "gamma": getattr(sweep_cfg, "gamma_rl", 0.99),
            "lr": getattr(sweep_cfg, "dqn_lr", 1e-3),
            "actor_lr": getattr(sweep_cfg, "actor_lr", 3e-4),
            "critic_lr": getattr(sweep_cfg, "critic_lr", 3e-4),
            "alpha_lr": getattr(sweep_cfg, "alpha_lr", 3e-4),
            "init_alpha": getattr(sweep_cfg, "init_alpha", 1.0),
            "buffer_size": getattr(sweep_cfg, "buffer_size", 50000),
            "batch_size": getattr(sweep_cfg, "rl_batch_size", 64),
            "hidden_dims": tuple(getattr(sweep_cfg, "hidden_dims", [64, 64])),
            "tau": getattr(sweep_cfg, "tau", 0.005),
            "train_freq": getattr(sweep_cfg, "train_freq", 4),
            "eval_freq": 5000,
            "epsilon_decay_fraction": 1.0
            / (1.0 + 1.0 / max(getattr(sweep_cfg, "epsilon_decay", 0.999), 0.9)),
        }

        full_training_timesteps = getattr(sweep_cfg, "full_training_timesteps", 200_000)
        finetune_timesteps = getattr(sweep_cfg, "finetune_timesteps", 5_000)

        # Demonstrator
        demo_policy_config = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=[256, 256],
            gamma=1.0,
        )

        demo_solver = MDPSolverApproximationExpectation(
            experiment_name=experiment_name,
            training_algorithm=demo_cfg.get("training_algorithm", "ppo"),
            T=env_cfg["T"],
            compute_variance=True,
            policy_config=demo_policy_config,
            training_timesteps=500_000,
        )

        demo = demonstrator.ContinuousDemonstrator(
            demo_env,
            demonstrator_name="Box2dDemonstrator",
            training_algorithm=demo_cfg.get("training_algorithm", "ppo"),
            T=env_cfg["T"],
            n_trajectories=demo_cfg.get("n_trajectories", 500),
            solver=demo_solver,
            hugging_face_repo=env_cfg.get("hugging_face_repo"),
        )
        log_memory("demonstrator_creation")

        learner_config = build_learner_config(sweep_cfg, _agent_type)

        if _agent_type == "mmd":
            train_and_evaluate_mmd(
                env=env,
                demo_env=demo_env,
                demo=demo,
                learner_config=learner_config,
                training_config=training_config,
                experiment_name=experiment_name,
                training_algorithm=training_algorithm,
                full_training_timesteps=full_training_timesteps,
                finetune_timesteps=finetune_timesteps,
                T=env_cfg["T"],
                n_trajectories_eval=env_cfg.get("n_trajectories_eval", 500),
                kernel_bandwidth=getattr(sweep_cfg, "kernel_bandwidth", None),
                tol_mmd=getattr(sweep_cfg, "tol_mmd", 0.01),
            )
        else:
            train_and_evaluate_jax(
                env=env,
                demo_env=demo_env,
                demo=demo,
                agent_type=_agent_type,
                learner_config=learner_config,
                training_config=training_config,
                experiment_name=experiment_name,
                training_algorithm=training_algorithm,
                full_training_timesteps=full_training_timesteps,
                finetune_timesteps=finetune_timesteps,
                T=env_cfg["T"],
                n_trajectories_eval=env_cfg.get("n_trajectories_eval", 500),
                lr_e=getattr(sweep_cfg, "lr_e", 0.1),
                lr_v=getattr(sweep_cfg, "lr_v", 0.05),
                lr_decay_rate_e=getattr(sweep_cfg, "lr_decay_rate", 0.95),
                lr_decay_rate_v=getattr(sweep_cfg, "lr_decay_rate_v", 0.9),
                alternate_every=getattr(sweep_cfg, "alternate_every", None),
                var_factor=getattr(sweep_cfg, "var_factor", 2),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for JAX-based Box2D environments"
    )
    parser.add_argument("config", help="Path to YAML config")
    parser.add_argument(
        "--agent-type",
        choices=["expectation", "variance", "mmd"],
        required=True,
        help="Which agent type to optimize",
    )
    parser.add_argument(
        "--sweep-id",
        default=None,
        help="Existing sweep ID to join (for parallel agents in separate terminals)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of runs for this agent (default: sweep_count from config)",
    )
    args = parser.parse_args()

    _yaml_config = load_config(args.config)
    _agent_type = args.agent_type

    adjusted_sweep = prepare_sweep_config(_yaml_config["sweep"], _agent_type)
    project = _yaml_config["wandb"]["project"]

    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        sweep_id = wandb.sweep(adjusted_sweep, project=f"{project}-{_agent_type}")
        print(f"Created sweep: {sweep_id}")
        print(f"To add parallel agents, run in other terminals:")
        print(f"  python -m tuning.sweep_box2d_jax {args.config} "
              f"--agent-type {_agent_type} --sweep-id {sweep_id}")

    count = args.count or _yaml_config["wandb"]["sweep_count"]
    wandb.agent(sweep_id, function=train, count=count, project=f"{project}-{_agent_type}")
