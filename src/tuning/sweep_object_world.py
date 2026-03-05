"""
Hyperparameter sweep for Object World environment.
Uses TabularLearner + MDPSolverExact, or TabularMMDLearner for MMD.

Usage:
    python -m tuning.sweep_object_world tuning/configs/object_world/grid6.yaml --agent-type expectation
    python -m tuning.sweep_object_world tuning/configs/object_world/grid6.yaml --agent-type variance
    python -m tuning.sweep_object_world tuning/configs/object_world/grid6.yaml --agent-type mmd
"""

import argparse
import wandb
import multiprocessing
import agents.demonstrator as demonstrator
from environments.object_world_environment import ObjectWorldEnvironment
from tuning.common import (
    load_config,
    prepare_sweep_config,
    build_optimizer_config,
    build_learner_config,
    train_and_evaluate_tabular,
    train_and_evaluate_tabular_mmd,
)

_yaml_config = None
_agent_type = None


def create_environment(env_cfg: dict) -> ObjectWorldEnvironment:
    return ObjectWorldEnvironment(
        {
            "theta": env_cfg.get("theta", [1.0, 1.0, -2.0]),
            "gamma": env_cfg["gamma"],
            "grid_size": env_cfg["grid_size"],
            "random_start": env_cfg.get("random_start", False),
            "continuous": env_cfg.get("continuous", False),
            "T": env_cfg["T"],
            "n_objects": env_cfg.get("n_objects", int(2 * env_cfg["grid_size"])),
        }
    )


def train():
    with wandb.init() as run:
        sweep_cfg = run.config
        env_cfg = _yaml_config["environment"]
        env_cfg["gamma"] = sweep_cfg["env_gamma"]
        env_cfg["T"] = sweep_cfg["env_T"]
        demo_cfg = _yaml_config["demonstrator"]

        env = create_environment(env_cfg)

        # wandb.log({"env_gamma": env.gamma}) # if not logged -> = 1.

        demo = demonstrator.ObjectWorldDemonstrator(
            env,
            demonstrator_name="ObjectWorldDemonstrator",
            T=env_cfg["T"],
            n_trajectories=demo_cfg.get("n_trajectories"),
        )

        learner_config = build_learner_config(sweep_cfg, _agent_type)

        if _agent_type == "mmd":
            train_and_evaluate_tabular_mmd(
                env=env,
                demo=demo,
                learner_config=learner_config,
                T=env_cfg["T"],
                n_trajectories_eval=env_cfg.get("n_trajectories_eval", 100),
                kernel_bandwidth=getattr(sweep_cfg, "kernel_bandwidth", None),
                tol_mmd=getattr(sweep_cfg, "tol_mmd", 0.01),
                show=False,
                store=False,
            )
        else:
            optimizer_config = build_optimizer_config(sweep_cfg, _agent_type)
            train_and_evaluate_tabular(
                env=env,
                demo=demo,
                agent_type=_agent_type,
                learner_config=learner_config,
                optimizer_config=optimizer_config,
                T=env_cfg["T"],
                n_trajectories_eval=env_cfg.get("n_trajectories_eval", 100),
                alternate_every=getattr(sweep_cfg, "alternate_every", None),
                var_factor=getattr(sweep_cfg, "var_factor", 2),
                show=False,
                store=False,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for Object World"
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
        print(f"  python -m tuning.sweep_object_world {args.config} "
              f"--agent-type {_agent_type} --sweep-id {sweep_id}")

    count = args.count or _yaml_config["wandb"]["sweep_count"]
    wandb.agent(sweep_id, function=train, count=count, project=f"{project}-{_agent_type}")
