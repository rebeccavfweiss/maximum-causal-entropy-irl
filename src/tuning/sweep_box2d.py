"""
Hyperparameter sweep for Box2D environments (LunarLander, BipedalWalker).
Uses ApproximateLearner + MDPSolverApproximation.

Usage:
    python -m tuning.sweep_box2d tuning/configs/box2d/lunarlander.yaml --agent-type expectation
    python -m tuning.sweep_box2d tuning/configs/box2d/lunarlander.yaml --agent-type variance
"""

import argparse
import torch
import wandb
import agents.demonstrator as demonstrator
from environments.box2d_environment import Box2DEnvironment
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


def create_environment(env_cfg: dict) -> Box2DEnvironment:
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

        env = create_environment(env_cfg)
        log_memory("env_config_creation")

        continuous = env_cfg.get("continuous", False)
        training_algorithm = getattr(
            sweep_cfg,
            "training_algorithm",
            "sac" if continuous else "dqn",
        )
        training_timesteps = getattr(sweep_cfg, "training_timesteps", 500_000)
        experiment_name = env_cfg["env_id"] + (
            "_continuous" if continuous else "_discrete"
        )

        # Build policy configs
        policy_config, policy_kwargs = build_policy_config(sweep_cfg)

        # Demonstrator policy config (separate — uses PPO with specific arch)
        demo_policy_config = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=[256, 256],
            gamma=1.0,
        )

        # Create demonstrator
        demo_solver = MDPSolverApproximationExpectation(
            experiment_name=experiment_name,
            training_algorithm=demo_cfg.get("training_algorithm", "ppo"),
            T=env_cfg["T"],
            compute_variance=True,
            policy_config=demo_policy_config,
            training_timesteps=training_timesteps,
        )

        demo = demonstrator.ContinuousDemonstrator(
            env,
            demonstrator_name="Box2dDemonstrator",
            training_algorithm=demo_cfg.get("training_algorithm", "ppo"),
            T=env_cfg["T"],
            n_trajectories=demo_cfg.get("n_trajectories", 500),
            solver=demo_solver,
            hugging_face_repo=env_cfg.get("hugging_face_repo"),
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
            n_trajectories_eval=env_cfg.get("n_trajectories_eval", 500),
            alternate_every=getattr(sweep_cfg, "alternate_every", None),
            var_factor=getattr(sweep_cfg, "var_factor", 2),
            show=False,
            store=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for Box2D environments"
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
