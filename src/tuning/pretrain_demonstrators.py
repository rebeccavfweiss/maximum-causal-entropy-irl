"""
Pre-train and cache demonstrators for all Object World configs.

Iterates over every YAML config in tuning/configs/object_world/,
creates (or loads) the environment with the fixed object layout,
trains the demonstrator, and saves policy + mu_demonstrator to disk.

Usage:
    python -m tuning.pretrain_demonstrators
    python -m tuning.pretrain_demonstrators --config-dir tuning/configs/object_world
    python -m tuning.pretrain_demonstrators --objects-config experiments/object_world/objects_train.json
"""

import argparse
import json
import os
from pathlib import Path

from environments.object_world_environment import ObjectWorldEnvironment
from tuning.common import load_config
from tuning.demonstrator_cache import (
    DEFAULT_CACHE_DIR,
    save_demonstrator,
    load_demonstrator,
)


def create_environment(env_cfg: dict, objects_config: list[dict] | None = None):
    cfg = {
        "theta": env_cfg.get("theta", [1.0, 1.0, -2.0]),
        "gamma": env_cfg.get("gamma", 1.0),
        "grid_size": env_cfg["grid_size"],
        "random_start": env_cfg.get("random_start", False),
        "continuous": env_cfg.get("continuous", False),
        "T": env_cfg.get("T", 20),
        "n_objects": env_cfg.get("n_objects", int(2 * env_cfg["grid_size"])),
    }
    if objects_config is not None:
        cfg["objects"] = objects_config
    return ObjectWorldEnvironment(cfg)


def pretrain_for_config(
    yaml_path: str,
    objects_config: list[dict] | None,
    cache_dir: Path,
):
    """Train and cache a demonstrator for a single YAML config."""
    yaml_config = load_config(yaml_path)
    env_cfg = yaml_config["environment"]
    demo_cfg = yaml_config.get("demonstrator", {})

    # Create or load object config
    if objects_config is None:
        env = create_environment(env_cfg)
        objects_config = env.export_objects_config()
        # Save for reproducibility
        os.makedirs("experiments/object_world", exist_ok=True)
        obj_path = f"experiments/object_world/objects_{Path(yaml_path).stem}.json"
        with open(obj_path, "w") as f:
            json.dump(objects_config, f, indent=2)
        print(f"Generated and saved object config to {obj_path}")
    else:
        env = create_environment(env_cfg, objects_config)

    demo_T = demo_cfg.get("T", env_cfg.get("T", 20))
    n_trajectories = demo_cfg.get("n_trajectories", None)
    theta = env_cfg.get("theta", [1.0, 1.0, -2.0])
    random_start = env_cfg.get("random_start", False)
    continuous = env_cfg.get("continuous", False)

    # Check if already cached
    existing = load_demonstrator(
        env, objects_config, demo_T, n_trajectories,
        theta=theta, random_start=random_start, continuous=continuous,
        cache_dir=cache_dir,
    )
    if existing is not None:
        print(f"  Demonstrator already cached for {yaml_path}, skipping.")
        return

    # Train
    from agents.demonstrator import ObjectWorldDemonstrator

    print(f"  Training demonstrator (T={demo_T}, n_trajectories={n_trajectories})...")
    demo = ObjectWorldDemonstrator(
        env,
        demonstrator_name="ObjectWorldDemonstrator",
        T=demo_T,
        n_trajectories=n_trajectories,
    )

    path = save_demonstrator(
        demo, objects_config,
        cache_dir=cache_dir,
        grid_size=env.grid_size,
        theta=theta,
        random_start=random_start,
        continuous=continuous,
    )
    print(f"  Saved demonstrator to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-train demonstrators for Object World configs"
    )
    parser.add_argument(
        "--config-dir",
        default="tuning/configs/object_world",
        help="Directory containing YAML config files",
    )
    parser.add_argument(
        "--objects-config",
        default=None,
        help="Path to JSON with fixed object placements (shared across all configs)",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Directory to store cached demonstrators",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    # Load objects config if provided
    objects_config = None
    if args.objects_config:
        with open(args.objects_config, "r") as f:
            objects_config = json.load(f)
        print(f"Using object config from {args.objects_config}")

    # Find all YAML configs
    config_dir = Path(args.config_dir)
    yaml_files = sorted(config_dir.glob("*.yaml"))

    if not yaml_files:
        print(f"No YAML configs found in {config_dir}")
        exit(1)

    print(f"Found {len(yaml_files)} config(s) in {config_dir}")

    for yaml_path in yaml_files:
        print(f"\nProcessing {yaml_path.name}...")
        pretrain_for_config(str(yaml_path), objects_config, cache_dir)

    print("\nDone.")
