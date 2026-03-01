"""
Sequential runner for multiple sweep configurations.

Runs wandb sweeps for each environment config, spawning each sweep as a
subprocess to ensure full memory cleanup between runs.

For each config, runs TWO sweeps by default: one for --agent-type expectation,
one for --agent-type variance. Use --agent-type to run only one.

Usage:
    python -m tuning.run_sweeps --env object_world --configs grid6.yaml grid8.yaml
    python -m tuning.run_sweeps --env object_world
    python -m tuning.run_sweeps --env all
    python -m tuning.run_sweeps --env box2d --agent-type expectation
"""

import argparse
import subprocess
import sys
from pathlib import Path

SWEEP_SCRIPTS = {
    "object_world": "tuning.sweep_object_world",
    "discrete_minigrid": "tuning.sweep_discrete_minigrid",
    "continuous_minigrid": "tuning.sweep_continuous_minigrid",
    "box2d": "tuning.sweep_box2d",
    "car_racing": "tuning.sweep_car_racing",
    "box2d_jax": "tuning.sweep_box2d_jax",
}

CONFIG_DIR = Path(__file__).parent / "configs"
SRC_DIR = Path(__file__).parent.parent  # src/ directory


def run_sweep(env_type: str, config_name: str, agent_type: str) -> None:
    """Run a single sweep as a subprocess."""
    script = SWEEP_SCRIPTS[env_type]
    config_path = CONFIG_DIR / env_type / config_name

    print(f"\n{'=' * 60}")
    print(f"Starting sweep: {env_type} / {config_name} / {agent_type}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            script,
            str(config_path),
            "--agent-type",
            agent_type,
        ],
        cwd=str(SRC_DIR),
    )

    if result.returncode != 0:
        print(
            f"WARNING: Sweep {env_type}/{config_name}/{agent_type} "
            f"exited with code {result.returncode}"
        )

    print(f"\nCompleted sweep: {env_type} / {config_name} / {agent_type}\n")


def get_all_configs(env_type: str) -> list[str]:
    """List all .yaml files in a config directory."""
    config_dir = CONFIG_DIR / env_type
    if not config_dir.exists():
        print(f"WARNING: Config directory not found: {config_dir}")
        return []
    return sorted(p.name for p in config_dir.glob("*.yaml"))


def main():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter sweeps sequentially"
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Environment type (object_world, discrete_minigrid, "
        "continuous_minigrid, box2d, car_racing) or 'all'",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Specific config files (default: all in directory)",
    )
    parser.add_argument(
        "--agent-type",
        choices=["expectation", "variance"],
        default=None,
        help="Run only one agent type (default: both)",
    )
    args = parser.parse_args()

    agent_types = (
        [args.agent_type] if args.agent_type else ["expectation", "variance"]
    )

    if args.env == "all":
        env_types = list(SWEEP_SCRIPTS.keys())
    else:
        if args.env not in SWEEP_SCRIPTS:
            print(
                f"ERROR: Unknown env type '{args.env}'. "
                f"Choose from: {list(SWEEP_SCRIPTS.keys())} or 'all'"
            )
            sys.exit(1)
        env_types = [args.env]

    for env_type in env_types:
        configs = args.configs or get_all_configs(env_type)
        if not configs:
            print(f"No configs found for {env_type}, skipping.")
            continue

        for config_name in configs:
            for agent_type in agent_types:
                run_sweep(env_type, config_name, agent_type)

    print("\nAll sweeps completed.")


if __name__ == "__main__":
    main()
