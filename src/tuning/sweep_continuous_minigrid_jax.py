"""
Hyperparameter sweep for JAX-based Continuous MiniGrid environments.
Uses JaxApproximateLearner + JaxSolver with warm-starting and CNN networks.

Usage:
    python -m tuning.sweep_continuous_minigrid_jax tuning/configs/continuous_minigrid_jax/doorkey5_jax.yaml --agent-type expectation
    python -m tuning.sweep_continuous_minigrid_jax tuning/configs/continuous_minigrid_jax/doorkey5_jax.yaml --agent-type variance
    python -m tuning.sweep_continuous_minigrid_jax tuning/configs/continuous_minigrid_jax/doorkey5_jax.yaml --agent-type mmd
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.30"

import argparse
import torch
import wandb
import agents.demonstrator as demonstrator
from environments.continuous_minigrid_jax_environment import ContinuousMinigridJaxEnvironment
from environments.continuous_minigrid_environment import ContinuousMinigridEnvironment
from solvers.MDP_solver_approximation import MDPSolverApproximationExpectation
from tuning.common import (
    load_config,
    prepare_sweep_config,
    build_learner_config,
    log_memory,
    train_and_evaluate_jax,
    train_and_evaluate_mmd,
)

import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th

_yaml_config = None
_agent_type = None


class DynamicMiniGridExtractor(BaseFeaturesExtractor):
    """CNN feature extractor for the SB3-based demonstrator."""

    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample_tensor = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_tensor).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))


def create_jax_environment(env_cfg: dict) -> ContinuousMinigridJaxEnvironment:
    return ContinuousMinigridJaxEnvironment(
        {
            "env_name": env_cfg["env_name"],
            "grid_size": env_cfg["grid_size"],
            "T": env_cfg["T"],
            "gamma": env_cfg.get("gamma", 1.0),
            "render_mode": "rgb_array",
            "seed": None,
        }
    )


def create_demo_environment(env_cfg: dict) -> ContinuousMinigridEnvironment:
    """Create SB3-based environment for the demonstrator."""
    return ContinuousMinigridEnvironment(
        {
            "env_name": env_cfg["env_name"],
            "grid_size": env_cfg["grid_size"],
            "T": env_cfg["T"],
            "gamma": env_cfg.get("gamma", 1.0),
            "render_mode": "rgb_array",
            "seed": None,
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

        experiment_name = env_cfg["env_name"] + "_jax"

        # Training config for JAX DQN
        training_config = {
            "gamma": env_cfg.get("gamma", 1.0),
            "lr": getattr(sweep_cfg, "dqn_lr", 1e-3),
            "buffer_size": getattr(sweep_cfg, "buffer_size", 50000),
            "batch_size": getattr(sweep_cfg, "rl_batch_size", 64),
            "hidden_dims": tuple(getattr(sweep_cfg, "hidden_dims", [64, 64])),
            "features_dim": getattr(sweep_cfg, "features_dim", 128),
            "tau": getattr(sweep_cfg, "tau", 0.005),
            "train_freq": getattr(sweep_cfg, "train_freq", 4),
            "eval_freq": 5000,
            "epsilon_decay_fraction": 1.0
            / (1.0 + 1.0 / max(getattr(sweep_cfg, "epsilon_decay", 0.999), 0.9)),
        }

        full_training_timesteps = getattr(sweep_cfg, "full_training_timesteps", 200_000)
        finetune_timesteps = getattr(sweep_cfg, "finetune_timesteps", 5_000)
        training_algorithm = getattr(sweep_cfg, "training_algorithm", "dqn")

        # Demonstrator (SB3-based)
        policy_kwargs = dict(
            features_extractor_class=DynamicMiniGridExtractor,
            features_extractor_kwargs=dict(features_dim=128),
        )
        policy_config = dict(
            policy="CnnPolicy",
            buffer_size=50000,
            tau=0.005,
            gamma=1.0,
            train_freq=5,
            device="auto",
        )

        demo_solver = MDPSolverApproximationExpectation(
            experiment_name=experiment_name,
            training_algorithm=demo_cfg.get("training_algorithm", "ppo"),
            T=env_cfg["T"],
            compute_variance=True,
            policy_config=policy_config,
            policy_kwargs=policy_kwargs,
            training_timesteps=demo_cfg.get("training_timesteps", 350000),
        )

        demo = demonstrator.ContinuousDemonstrator(
            demo_env,
            demonstrator_name="MinigridDemonstrator",
            training_algorithm=demo_cfg.get("training_algorithm", "ppo"),
            T=env_cfg["T"],
            n_trajectories=demo_cfg.get("n_trajectories", 150),
            solver=demo_solver,
            policy_kwargs=policy_kwargs,
            time_steps=demo_cfg.get("time_steps", 7_500_000),
            policy_type="CnnPolicy",
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
                n_trajectories_eval=env_cfg.get("n_trajectories_eval", 150),
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
                n_trajectories_eval=env_cfg.get("n_trajectories_eval", 150),
                lr_e=getattr(sweep_cfg, "lr_e", 0.1),
                lr_v=getattr(sweep_cfg, "lr_v", 0.05),
                lr_decay_rate_e=getattr(sweep_cfg, "lr_decay_rate", 0.95),
                lr_decay_rate_v=getattr(sweep_cfg, "lr_decay_rate_v", 0.9),
                alternate_every=getattr(sweep_cfg, "alternate_every", None),
                var_factor=getattr(sweep_cfg, "var_factor", 2),
            )


if __name__ == "__main__":
    import minigrid  # noqa: F401 — registers MiniGrid envs

    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for JAX-based Continuous MiniGrid"
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
        print(f"  python -m tuning.sweep_continuous_minigrid_jax {args.config} "
              f"--agent-type {_agent_type} --sweep-id {sweep_id}")

    count = args.count or _yaml_config["wandb"]["sweep_count"]
    wandb.agent(sweep_id, function=train, count=count, project=f"{project}-{_agent_type}")
