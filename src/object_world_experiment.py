import json
import agents.learner as learner
from environments.object_world_environment import ObjectWorldEnvironment
from tuning.demonstrator_cache import load_or_train_demonstrator
import solvers.MDP_solver_exact as MDPSolver
import numpy as np
import pandas as pd
from multiprocessing import Pool
import wandb
import matplotlib.pylab as plt
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, CyclicLR
from torch.optim import Adamax, Adam, SGD


def create_objectworld_env(
    gamma: float = 1.0,
    grid_size: int = 32,
    T: int = 50,
    random_start: bool = True,
    continuous: bool = False,
    n_objects: int = 2,
    objects: list[dict] = None,
):

    config_env = {
        "theta": [
            1.0,
            1.0,
            -2.0,
        ],  # just any thing because the base class needs this variable, not used here
        "gamma": gamma,
        "grid_size": grid_size,
        "random_start": random_start,
        "continuous": continuous,
        "T": T,
        "n_objects": n_objects,
    }

    if objects is not None:
        config_env["objects"] = objects

    env = ObjectWorldEnvironment(config_env)

    return env


def create_config_learner():
    config_default_learner = {
        "tol_exp": 0.05,
        "tol_var": 7.5,
        "miniter": 1,
        "maxiter": 5_000,
    }

    return config_default_learner


def run_experiment(args):
    np.random.seed()
    n_trajectories, run, objects_train = args
    show = False
    store = False
    experiment_name = "object-world"
    T = 50
    demo_T = 8
    grid_size = 32
    gamma = 1.0
    random_start = True
    continuous = False
    n_objects = 18
    config_default_learner = create_config_learner()
    learning_rate = {
        "scheduler": ReduceLROnPlateau,
        "scheduler_kwargs": {"min_lr": 0.0001, "factor": 0.5},
    }
    learning_rate_e ={
        "scheduler": ReduceLROnPlateau,
        "scheduler_kwargs": {"min_lr": 0.0001, "factor":0.5},
    }
    learning_rate_v = {
        "scheduler": ReduceLROnPlateau,
        "scheduler_kwargs": {"min_lr": 0.0001, "factor": 0.5},
    }
    optimizer = Adam
    optimizer_e = SGD
    optimizer_v = Adamax
    optimizer_kwargs = {"lr": 0.025198630693598355, "weight_decay":0.001}
    optimizer_e_kwargs = {"lr": 0.017406304685365078, "weight_decay":0.001}
    optimizer_v_kwargs = {"lr": 0.05738768503625415, "eps": 1e-7, "weight_decay": 0.005}
    alternate_every = None
    var_factor = 3

    wandb.init(
        project=f"mceirl-{experiment_name}",
        name=f"{experiment_name}-g{grid_size}-r{run}-n{n_trajectories}"
        + ("-random" if random_start else "")
        + ("-cont_features" if continuous else "-discrete_features"),
        config={
            "T": T,
            "grid_size": grid_size,
            "run_id": run,
            "random_start": random_start,
            "n_trajectories": n_trajectories,
            "continuous": continuous,
            "alternate_every": alternate_every,
        },
    )

    env = create_objectworld_env(
        gamma=gamma,
        T=T,
        grid_size=grid_size,
        random_start=random_start,
        continuous=continuous,
        n_objects=n_objects,
        objects=objects_train,
    )
    config_default_learner = create_config_learner()

    reward_matrix = env.reward.reshape(env.grid_size, env.grid_size)

    fig = plt.figure()
    plt.pcolor(reward_matrix)
    plt.colorbar()
    plt.title("True reward")
    if show:
        plt.show()
    if store:
        plt.savefig(
            Path("plots")
            / "object_world_environment"
            / f"n{n_trajectories}_r{run}_real_rewards.jpg",
            format="jpg",
        )

    # load cached demonstrator or train a new one
    demo = load_or_train_demonstrator(
        env,
        objects_config=objects_train,
        demo_T=demo_T,
        n_trajectories=None,
        theta=[1.0, 1.0, -2.0],
        random_start=random_start,
        continuous=continuous,
    )

    reward_demonstrator = env.compute_true_reward_for_agent(demo, n_trajectories, T)

    wandb.log(
        {
            "demonstrator_expected_value": demo.mu_demonstrator[0],
            "demonstrator_variance": demo.mu_demonstrator[1],
            "demonstrator_reward": reward_demonstrator,
            "theta_*": env.theta_reward,
        }
    )

    # agent using only expectation matching
    agent_expectation = learner.TabularLearner(
        env,
        demo.mu_demonstrator,
        config_default_learner,
        agent_name=f"AgentExpectation",
        solver=MDPSolver.MDPSolverExactExpectation(T),
        learning_rate_e=learning_rate,
        optimizer_e=optimizer,
        optimizer_e_kwargs=optimizer_kwargs,
    )
    iter_expectation, time_expectation = agent_expectation.batch_MCE()
    agent_expectation.compute_and_draw(show, store, 4)
    reward_expectation = env.compute_true_reward_for_agent(
        agent_expectation, n_trajectories, T
    )

    learned_reward_grid_expectation = agent_expectation.rewards[-1].reshape(
        env.grid_size, env.grid_size
    )
    lrge_min = learned_reward_grid_expectation.min()
    lrge_max = learned_reward_grid_expectation.max()
    learned_reward_grid_expectation = (
        (learned_reward_grid_expectation - lrge_min) / (lrge_max - lrge_min)
    ) * 2 - 1

    wandb.log(
        {
            "reward_expectation": reward_expectation,
            "reward_diff_expectation": reward_expectation - reward_demonstrator,
            "iterations_expectation": iter_expectation,
            "time_total_expectation": sum(time_expectation),
            "time_avg_per_iter_expectation": np.mean(time_expectation),
            "reward_grid_diff_expectation": learned_reward_grid_expectation
            - reward_matrix,
            "reward_grid_diff_norm_expectation": np.linalg.norm(
                learned_reward_grid_expectation - reward_matrix
            ),
        }
    )

    # agent that also matches variances
    agent_variance = learner.TabularLearner(
        env,
        demo.mu_demonstrator,
        config_default_learner,
        agent_name=f"AgentVariance",
        solver=MDPSolver.MDPSolverExactVariance(T),
        learning_rate_e=learning_rate_e,
        learning_rate_v=learning_rate_v,
        optimizer_e=optimizer_e,
        optimizer_v=optimizer_v,
        optimizer_e_kwargs=optimizer_e_kwargs,
        optimizer_v_kwargs=optimizer_v_kwargs,
    )
    iter_variance, time_variance = agent_variance.batch_MCE(
        alternate_every=alternate_every, var_factor=var_factor
    )
    agent_variance.compute_and_draw(show, store, 7)
    reward_variance = env.compute_true_reward_for_agent(
        agent_variance, n_trajectories, T
    )

    learned_reward_grid_variance = agent_variance.rewards[-1].reshape(
        env.grid_size, env.grid_size
    )
    lrgv_min = learned_reward_grid_variance.min()
    lrgv_max = learned_reward_grid_variance.max()
    learned_reward_grid_variance = (
        (learned_reward_grid_variance - lrgv_min) / (lrgv_max - lrgv_min)
    ) * 2 - 1

    wandb.log(
        {
            "reward_variance": reward_variance,
            "reward_diff_variance": reward_variance - reward_demonstrator,
            "iterations_variance": iter_variance,
            "time_total_variance": sum(time_variance),
            "time_avg_per_iter_variance": np.mean(time_variance),
            "reward_grid_diff_variance": learned_reward_grid_variance - reward_matrix,
            "reward_grid_diff_norm_variance": np.linalg.norm(
                learned_reward_grid_variance - reward_matrix
            ),
        }
    )

    wandb.finish()


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Object World experiment")
    parser.add_argument(
        "--objects-train", default=None,
        help="Path to JSON with fixed training object config",
    )
    cli_args = parser.parse_args()

    # Load or generate fixed object configs
    os.makedirs("experiments/object_world", exist_ok=True)

    if cli_args.objects_train:
        with open(cli_args.objects_train, "r") as f:
            objects_train = json.load(f)
    else:
        # Generate and save a training config
        seed_env = create_objectworld_env()
        objects_train = seed_env.export_objects_config()
        with open("experiments/object_world/objects_train.json", "w") as f:
            json.dump(objects_train, f, indent=2)
        print("Saved training object config to experiments/object_world/objects_train.json")

    tasks = []
    for n_trajectories in [1, 10, 50, 100, 200, 500, 1000, 2500, 5000]:
        for run in range(10):
            tasks.append((n_trajectories, run, objects_train))

    with Pool(processes=5) as pool:
        results = pool.map(run_experiment, tasks)
