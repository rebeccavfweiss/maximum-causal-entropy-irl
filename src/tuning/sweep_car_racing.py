"""
Hyperparameter sweep for Car Racing environment.
Uses ApproximateLearner + MDPSolverApproximation with heuristic rewards.

Usage:
    python -m tuning.sweep_car_racing tuning/configs/car_racing/discrete.yaml --agent-type expectation
    python -m tuning.sweep_car_racing tuning/configs/car_racing/discrete.yaml --agent-type variance
"""

import argparse
import numpy as np
import wandb
import agents.demonstrator as demonstrator
from environments.car_racing_environment import CarRacingEnvironment
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


def temporal_diff_matrix(num_frames: int, frame_size: int) -> np.ndarray:
    """
    Construct a quadratic penalty matrix D such that:
    R(s) = s^T D s = sum_i ||f_i - f_{i-1}||^2
    """
    size = num_frames * frame_size
    D = np.zeros((size, size))
    I = np.eye(frame_size)

    for i in range(1, num_frames):
        a = (i - 1) * frame_size
        b = i * frame_size
        D[a : a + frame_size, a : a + frame_size] += I
        D[b : b + frame_size, b : b + frame_size] += I
        D[a : a + frame_size, b : b + frame_size] -= I
        D[b : b + frame_size, a : a + frame_size] -= I

    return D


def compute_heuristics(
    width: int, height: int, weight_forward: float, weight_speed_limitation: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute heuristic theta_e and theta_v for car racing."""
    y = np.linspace(-1, 1, width)
    x = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dist_from_center = np.sqrt(X**2 + Y**2) - 0.5 * X

    center_mask = 1.0 - dist_from_center
    center_mask = np.clip(center_mask, 0, 0.9)
    center_mask /= 255.0

    D = temporal_diff_matrix(4, width * height)

    h_theta_e = -np.tile(center_mask.flatten(), 4)
    h_theta_v = (
        np.diag(h_theta_e) + weight_forward * D - weight_speed_limitation * D.dot(D)
    )

    return h_theta_e, h_theta_v


def create_environment(env_cfg: dict) -> CarRacingEnvironment:
    return CarRacingEnvironment(
        {
            "gamma": env_cfg.get("gamma", 1.0),
            "lap_complete_percent": env_cfg.get("lap_complete_percent", 0.33),
            "T": env_cfg["T"],
            "n_frames": env_cfg.get("n_frames", 4),
            "width": env_cfg.get("width", 84),
            "height": env_cfg.get("height", 84),
            "continuous_actions": env_cfg.get("continuous_actions", False),
        }
    )


def train():
    with wandb.init() as run:
        sweep_cfg = run.config
        env_cfg = _yaml_config["environment"]
        demo_cfg = _yaml_config["demonstrator"]

        env = create_environment(env_cfg)
        log_memory("env_config_creation")

        continuous_actions = env_cfg.get("continuous_actions", False)
        training_algorithm = getattr(
            sweep_cfg,
            "training_algorithm",
            "sac" if continuous_actions else "dqn",
        )
        training_timesteps = getattr(sweep_cfg, "training_timesteps", 350_000)
        experiment_name = "car_racing" + (
            "_continuous" if continuous_actions else "_discrete"
        )

        # Compute heuristic reward matrices
        weight_forward = getattr(sweep_cfg, "weight_forward", 0.1)
        weight_speed_limitation = getattr(sweep_cfg, "weight_speed_limitation", 0.05)
        heuristic_theta_e, heuristic_theta_v = compute_heuristics(
            env.frame_width,
            env.frame_height,
            weight_forward,
            weight_speed_limitation,
        )

        policy_config, policy_kwargs = build_policy_config(sweep_cfg)

        # Create demonstrator
        demo_training_alg = demo_cfg.get(
            "training_algorithm", "ppo" if continuous_actions else "dqn"
        )
        demo_solver = MDPSolverApproximationExpectation(
            experiment_name=experiment_name,
            training_algorithm=demo_training_alg,
            T=env_cfg["T"],
            compute_variance=True,
            policy_config=policy_config,
            training_timesteps=training_timesteps,
        )

        demo = demonstrator.ContinuousDemonstrator(
            env,
            demonstrator_name="CarRacingDemonstrator",
            training_algorithm=demo_training_alg,
            T=env_cfg["T"],
            n_trajectories=demo_cfg.get("n_trajectories", 150),
            solver=demo_solver,
        )
        log_memory("demonstrator_creation")

        optimizer_config = build_optimizer_config(sweep_cfg, _agent_type)
        learner_config = build_learner_config(sweep_cfg, _agent_type)

        # Pass heuristics based on agent type
        h_theta_e = heuristic_theta_e
        h_theta_v = heuristic_theta_v if _agent_type == "variance" else None

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
            early_stop_window=getattr(sweep_cfg, "early_stop_window", 200),
            heuristic_theta_e=h_theta_e,
            heuristic_theta_v=h_theta_v,
            show=False,
            store=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for Car Racing"
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
