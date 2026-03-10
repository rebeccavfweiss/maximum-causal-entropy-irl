"""
Hyperparameter sweep for CliffWalking environment.
Uses TabularLearner + MDPSolverExact for expectation/variance matching.
Always uses non-one-hot (3D) features.

success_rate, gamma, and T are fixed per config (T = 50 * int(1/success_rate)).

Usage:
    python -m tuning.sweep_cliff_walking tuning/configs/cliff_walking/default.yaml --agent-type expectation
    python -m tuning.sweep_cliff_walking tuning/configs/cliff_walking/default.yaml --agent-type variance
"""

import argparse
import numpy as np
import wandb
import agents.demonstrator as demonstrator
from environments.cliff_walking_environment import CliffWalkingEnvironment
from tuning.common import (
    load_config,
    prepare_sweep_config,
    build_optimizer_config,
    build_learner_config,
    train_and_evaluate_tabular,
)

_yaml_config = None
_agent_type = None


def create_environment(env_cfg: dict) -> CliffWalkingEnvironment:
    n_states = 49  # 48 grid + 1 absorbing
    theta = np.zeros(n_states)

    for s in range(48):
        theta[s] = -1.0

    for s in CliffWalkingEnvironment.CLIFF_STATES:
        theta[s] = -100.0

    theta[CliffWalkingEnvironment.START_STATE] = 0.0
    theta[48] = 0.0

    success_rate = env_cfg["success_rate"]
    T = int(50 * int(1 / success_rate))

    return CliffWalkingEnvironment(
        {
            "theta": theta,
            "gamma": env_cfg["gamma"],
            "success_rate": success_rate,
            "T": T,
            "one_hot_features": False,
        }
    ), T


def train():
    with wandb.init() as run:
        sweep_cfg = run.config
        env_cfg = _yaml_config["environment"]

        env, T = create_environment(env_cfg)

        wandb.log(
            {
                "success_rate": env.success_rate,
                "env_T": T,
                "env_gamma": env.gamma,
            }
        )

        demo = demonstrator.CliffWalkingDemonstrator(
            env,
            demonstrator_name="CliffWalkingDemonstrator",
            T=T,
        )

        learner_config = build_learner_config(sweep_cfg, _agent_type)
        optimizer_config = build_optimizer_config(sweep_cfg, _agent_type)

        train_and_evaluate_tabular(
            env=env,
            demo=demo,
            agent_type=_agent_type,
            learner_config=learner_config,
            optimizer_config=optimizer_config,
            T=T,
            n_trajectories_eval=None,
            alternate_every=getattr(sweep_cfg, "alternate_every", None),
            var_factor=getattr(sweep_cfg, "var_factor", 2),
            show=False,
            store=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for CliffWalking"
    )
    parser.add_argument("config", help="Path to YAML config")
    parser.add_argument(
        "--agent-type",
        choices=["expectation", "variance"],
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
        print(
            f"  python -m tuning.sweep_cliff_walking {args.config} "
            f"--agent-type {_agent_type} --sweep-id {sweep_id}"
        )

    count = args.count or _yaml_config["wandb"]["sweep_count"]
    wandb.agent(
        sweep_id, function=train, count=count, project=f"{project}-{_agent_type}"
    )
