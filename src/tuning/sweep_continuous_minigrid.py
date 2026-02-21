"""
Hyperparameter sweep for Continuous MiniGrid environment.
Uses ApproximateLearner + MDPSolverApproximation.

Usage:
    python -m tuning.sweep_continuous_minigrid tuning/configs/continuous_minigrid/doorkey5.yaml --agent-type expectation
    python -m tuning.sweep_continuous_minigrid tuning/configs/continuous_minigrid/doorkey5.yaml --agent-type variance
"""

import argparse
import wandb
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import agents.demonstrator as demonstrator
from environments.continuous_minigrid_environment import ContinuousMinigridEnvironment
from solvers.MDP_solver_approximation import MDPSolverApproximationExpectation
from tuning.common import (
    load_config,
    prepare_sweep_config,
    build_optimizer_config,
    build_learner_config,
    build_policy_config,
    train_and_evaluate_approximate,
    log_memory,
)

_yaml_config = None
_agent_type = None


class DynamicMiniGridExtractor(BaseFeaturesExtractor):
    """CNN feature extractor that dynamically computes output shape."""

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


def create_environment(env_cfg: dict) -> ContinuousMinigridEnvironment:
    return ContinuousMinigridEnvironment(
        {
            "gamma": env_cfg.get("gamma", 1.0),
            "env_name": env_cfg["env_name"],
            "render_mode": "rgb_array",
            "grid_size": env_cfg["grid_size"],
            "seed": None,
            "T": env_cfg["T"],
        }
    )


def train():
    with wandb.init() as run:
        sweep_cfg = run.config
        env_cfg = _yaml_config["environment"]
        demo_cfg = _yaml_config["demonstrator"]

        env = create_environment(env_cfg)
        log_memory("env_config_creation")

        policy_config, policy_kwargs = build_policy_config(
            sweep_cfg, features_extractor_class=DynamicMiniGridExtractor
        )

        training_algorithm = getattr(sweep_cfg, "training_algorithm", "dqn")
        training_timesteps = getattr(sweep_cfg, "training_timesteps", 350000)
        experiment_name = env_cfg["env_name"]

        # Create demonstrator
        demo_solver = MDPSolverApproximationExpectation(
            experiment_name=experiment_name,
            training_algorithm=demo_cfg.get("training_algorithm", "ppo"),
            T=env_cfg["T"],
            compute_variance=True,
            policy_config=policy_config,
            policy_kwargs=policy_kwargs,
            training_timesteps=training_timesteps,
        )

        demo = demonstrator.ContinuousDemonstrator(
            env,
            demonstrator_name="MinigridDemonstrator",
            T=env_cfg["T"],
            n_trajectories=demo_cfg.get("n_trajectories", 150),
            training_algorithm=demo_cfg.get("training_algorithm", "ppo"),
            solver=demo_solver,
            policy_kwargs=policy_kwargs,
            time_steps=demo_cfg.get("time_steps", 7_500_000),
        )
        log_memory("demonstrator_creation")

        optimizer_config = build_optimizer_config(sweep_cfg, _agent_type)
        learner_config = build_learner_config(sweep_cfg, _agent_type)

        train_and_evaluate_approximate(
            env=env,
            demo=demo,
            agent_type=_agent_type,
            learner_config=learner_config,
            optimizer_config=optimizer_config,
            policy_config=policy_config,
            policy_kwargs=policy_kwargs,
            experiment_name=experiment_name,
            training_algorithm=training_algorithm,
            training_timesteps=training_timesteps,
            T=env_cfg["T"],
            n_trajectories_eval=env_cfg.get("n_trajectories_eval", 150),
            alternate_every=getattr(sweep_cfg, "alternate_every", None),
            var_factor=getattr(sweep_cfg, "var_factor", 2),
            show=False,
            store=True,
        )


if __name__ == "__main__":
    import minigrid  # noqa: F401 — registers MiniGrid envs

    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for Continuous MiniGrid"
    )
    parser.add_argument("config", help="Path to YAML config")
    parser.add_argument(
        "--agent-type",
        choices=["expectation", "variance"],
        required=True,
        help="Which agent type to optimize",
    )
    args = parser.parse_args()

    _yaml_config = load_config(args.config)
    _agent_type = args.agent_type

    adjusted_sweep = prepare_sweep_config(_yaml_config["sweep"], _agent_type)
    project = _yaml_config["wandb"]["project"]

    sweep_id = wandb.sweep(adjusted_sweep, project=f"{project}-{_agent_type}")
    wandb.agent(
        sweep_id,
        function=train,
        count=_yaml_config["wandb"]["sweep_count"],
    )
