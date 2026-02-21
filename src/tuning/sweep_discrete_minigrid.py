"""
Hyperparameter sweep for Discrete MiniGrid environment.
Uses TabularLearner + MDPSolverExact.

Usage:
    python -m tuning.sweep_discrete_minigrid tuning/configs/discrete_minigrid/grid9.yaml --agent-type expectation
    python -m tuning.sweep_discrete_minigrid tuning/configs/discrete_minigrid/grid9.yaml --agent-type variance
"""

import argparse
import numpy as np
import multiprocessing
import wandb
from random import randint
import agents.demonstrator as demonstrator
from environments.discrete_minigrid_environment import CrossingMiniGridEnvironment
from tuning.common import (
    load_config,
    prepare_sweep_config,
    build_optimizer_config,
    build_learner_config,
    train_and_evaluate_tabular,
)

_yaml_config = None
_agent_type = None


def create_environment(env_cfg: dict) -> CrossingMiniGridEnvironment:
    grid_size = env_cfg["grid_size"]

    theta = np.eye(2 * grid_size - 2, 2 * grid_size - 2)
    theta[0, 0] = theta[1, 1] = -1.0
    theta[-2, -2] = 10.0
    theta[-1, -1] = -10.0

    return CrossingMiniGridEnvironment(
        {
            "theta": theta,
            "gamma": env_cfg.get("gamma", 1.0),
            "env_name": env_cfg.get("env_name", "MiniGrid-LavaCrossingS9N1-v0"),
            "render_mode": "rgb_array",
            "grid_size": grid_size,
            "seed": randint(1, 100),
        }
    )


def train():
    with wandb.init() as run:
        sweep_cfg = run.config
        env_cfg = _yaml_config["environment"]

        env = create_environment(env_cfg)

        demo = demonstrator.CrossingMinigridDemonstrator(
            env,
            demonstrator_name="MiniGridDemonstrator",
            T=env_cfg["T"],
        )

        optimizer_config = build_optimizer_config(sweep_cfg, _agent_type)
        learner_config = build_learner_config(sweep_cfg, _agent_type)

        train_and_evaluate_tabular(
            env=env,
            demo=demo,
            agent_type=_agent_type,
            learner_config=learner_config,
            optimizer_config=optimizer_config,
            T=env_cfg["T"],
            n_trajectories_eval=env_cfg.get("n_trajectories_eval"),
            alternate_every=getattr(sweep_cfg, "alternate_every", None),
            var_factor=getattr(sweep_cfg, "var_factor", 2),
            show=False,
            store=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for Discrete MiniGrid"
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
    # wandb.agent(
    #     sweep_id,
    #     function=train,
    #     count=_yaml_config["wandb"]["sweep_count"],
    # )

    num_agents = 5 

    processes = []
    for i in range(num_agents):
        # We use a helper to call wandb.agent in a separate process
        p = multiprocessing.Process(
            target=wandb.agent, 
            args=(sweep_id,), 
            kwargs={'function': train, 'count': int(_yaml_config["wandb"]["sweep_count"]/num_agents)} # Each agent does 5 runs
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

