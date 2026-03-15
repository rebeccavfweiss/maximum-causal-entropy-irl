"""
Hyperparameter sweep for Object World environment.
Uses TabularLearner + MDPSolverExact, or TabularMMDLearner for MMD.

A single fixed object configuration is generated (or loaded) before the sweep
starts, so that all hyperparameter combinations are evaluated on the same
environment layout.

Usage:
    python -m tuning.sweep_object_world tuning/configs/object_world/grid6.yaml --agent-type expectation
    python -m tuning.sweep_object_world tuning/configs/object_world/grid6.yaml --agent-type variance
    python -m tuning.sweep_object_world tuning/configs/object_world/grid6.yaml --agent-type mmd

    # Reuse a previously saved object config:
    python -m tuning.sweep_object_world tuning/configs/object_world/grid6.yaml --agent-type variance \
        --objects-config experiments/object_world/objects_train.json
"""

import argparse
import json
import wandb
from environments.object_world_environment import ObjectWorldEnvironment
from tuning.common import (
    load_config,
    prepare_sweep_config,
    build_optimizer_config,
    build_learner_config,
    train_and_evaluate_tabular,
    train_and_evaluate_tabular_mmd,
)
from tuning.demonstrator_cache import load_or_train_demonstrator

_yaml_config = None
_agent_type = None
_objects_config = None  # fixed object placement shared across all sweep runs


def create_environment(env_cfg: dict) -> ObjectWorldEnvironment:
    cfg = {
        "theta": env_cfg.get("theta", [1.0, 1.0, -2.0]),
        "gamma": env_cfg["gamma"],
        "grid_size": env_cfg["grid_size"],
        "random_start": env_cfg.get("random_start", False),
        "continuous": env_cfg.get("continuous", False),
        "T": env_cfg["T"],
        "n_objects": env_cfg.get("n_objects", int(2 * env_cfg["grid_size"])),
    }
    if _objects_config is not None:
        cfg["objects"] = _objects_config
    return ObjectWorldEnvironment(cfg)


def train():
    with wandb.init() as run:
        sweep_cfg = run.config
        env_cfg = _yaml_config["environment"]
        env_cfg["gamma"] = sweep_cfg["env_gamma"]
        env_cfg["T"] = sweep_cfg["env_T"]
        demo_cfg = _yaml_config["demonstrator"]

        env = create_environment(env_cfg)

        demo_T = demo_cfg.get("T", env_cfg["T"])
        demo = load_or_train_demonstrator(
            env,
            objects_config=_objects_config,
            demo_T=demo_T,
            n_trajectories=demo_cfg.get("n_trajectories"),
            theta=env_cfg.get("theta", [1.0, 1.0, -2.0]),
            random_start=env_cfg.get("random_start", False),
            continuous=env_cfg.get("continuous", False),
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
    parser.add_argument(
        "--objects-config",
        default=None,
        help="Path to JSON file with fixed object placements (generated if not provided)",
    )
    args = parser.parse_args()

    _yaml_config = load_config(args.config)
    _agent_type = args.agent_type

    # Generate or load a fixed object configuration for the entire sweep
    if args.objects_config:
        with open(args.objects_config, "r") as f:
            _objects_config = json.load(f)
        print(f"Loaded object config from {args.objects_config}")
    else:
        # Create one env to generate random objects, then export and reuse
        env_cfg = _yaml_config["environment"]
        seed_env = ObjectWorldEnvironment(
            {
                "theta": env_cfg.get("theta", [1.0, 1.0, -2.0]),
                "gamma": env_cfg.get("gamma", 1.0),
                "grid_size": env_cfg["grid_size"],
                "random_start": env_cfg.get("random_start", False),
                "continuous": env_cfg.get("continuous", False),
                "T": env_cfg.get("T", 20),
                "n_objects": env_cfg.get("n_objects", int(2 * env_cfg["grid_size"])),
            }
        )
        _objects_config = seed_env.export_objects_config()

        # Save for reproducibility
        import os
        os.makedirs("experiments/object_world", exist_ok=True)
        config_path = f"experiments/object_world/objects_sweep_{_agent_type}.json"
        with open(config_path, "w") as f:
            json.dump(_objects_config, f, indent=2)
        print(f"Generated and saved object config to {config_path}")

    adjusted_sweep = prepare_sweep_config(_yaml_config["sweep"], _agent_type)
    project = _yaml_config["wandb"]["project"]

    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        sweep_id = wandb.sweep(adjusted_sweep, project=f"{project}-{_agent_type}")
        print(f"Created sweep: {sweep_id}")
        print(f"To add parallel agents, run in other terminals:")
        print(
            f"  python -m tuning.sweep_object_world {args.config} "
            f"--agent-type {_agent_type} --sweep-id {sweep_id} "
            f"--objects-config experiments/object_world/objects_sweep_{_agent_type}.json"
        )

    count = args.count or _yaml_config["wandb"]["sweep_count"]
    wandb.agent(
        sweep_id, function=train, count=count, project=f"{project}-{_agent_type}"
    )
