import agents.learner as learner
import agents.demonstrator as demonstrator
from environments.object_world_environment import ObjectWorldEnvironment
import solvers.MDP_solver_exact as MDPSolver
import numpy as np
import pandas as pd
from multiprocessing import Pool
import wandb
import matplotlib.pylab as plt
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, CyclicLR
from torch.optim import Adamax, Adam


def create_objectworld_env(
    gamma: float = 1.0,
    grid_size: int = 32,
    T: int = 50,
    random_start: bool = True,
    continuous: bool = False,
    n_objects: int = 2,
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

    env = ObjectWorldEnvironment(config_env)

    return env


def create_config_learner():
    config_default_learner = {
        "tol_exp": 0.005,
        "tol_var": 0.1,
        "miniter": 1,
        "maxiter": 5_000,
    }

    return config_default_learner


def run_experiment(args):
    np.random.seed()
    n_trajectories, run = args
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
    learning_rate_e = {
        "scheduler": LambdaLR,
        "scheduler_kwargs": {
            "lr_lambda": lambda step: max(0.95 ** np.log(step + 1), 0.001)
        },
    }
    learning_rate_v = {
        "scheduler": ReduceLROnPlateau,
        "scheduler_kwargs": {"min_lr": 0.0001, "factor": 0.5},
    }
    optimizer = Adam
    optimizer_e = Adam
    optimizer_v = Adamax
    optimizer_kwargs = {"lr": 0.025198630693598355, "weight_decay": 0.001}
    optimizer_e_kwargs = {"lr": 0.013481979369633936, "weight_decay": 0.001}
    optimizer_v_kwargs = {"lr": 0.06503275998396742, "eps": 1e-7, "weight_decay": 0.005}
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

    # create demonstrator with shorter trajectory length
    demo = demonstrator.ObjectWorldDemonstrator(
        env,
        demonstrator_name="ObjectWorldDemonstrator",
        T=demo_T,
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
        optimizer_e=optimizer_e,
        optimizer_e_kwargs=optimizer_e_kwargs,
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
        optimizer_e=optimizer_v,
        optimizer_v=optimizer_v,
        optimizer_e_kwargs=optimizer_v_kwargs,
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

    # Evaluate on a fresh environment with newly randomized object placement
    env_new = create_objectworld_env(
        gamma=gamma,
        T=T,
        grid_size=grid_size,
        random_start=random_start,
        continuous=continuous,
        n_objects=n_objects,
    )

    reward_demonstrator_new = env_new.compute_true_reward_for_agent(
        demo, n_trajectories, T
    )
    reward_expectation_new = env_new.compute_true_reward_for_agent(
        agent_expectation, n_trajectories, T
    )
    reward_variance_new = env_new.compute_true_reward_for_agent(
        agent_variance, n_trajectories, T
    )

    wandb.log(
        {
            "reward_demonstrator_new_env": reward_demonstrator_new,
            "reward_expectation_new_env": reward_expectation_new,
            "reward_diff_expectation_new_env": reward_expectation_new
            - reward_demonstrator_new,
            "reward_variance_new_env": reward_variance_new,
            "reward_diff_variance_new_env": reward_variance_new
            - reward_demonstrator_new,
        }
    )

    wandb.finish()


if __name__ == "__main__":

    tasks = []
    for n_trajectories in [1, 10, 50, 100, 200, 500, 1000, 2500, 5000]:
        for run in range(10):
            tasks.append((n_trajectories, run))

    with Pool(processes=5) as pool:
        results = pool.map(run_experiment, tasks)
