import agents.learner as learner
import agents.demonstrator as demonstrator
from environments.object_world_environment import ObjectWorldEnvironment
import MDP_solver_exact as MDPSolver
import numpy as np
import pandas as pd
from multiprocessing import Pool
import wandb
import matplotlib.pylab as plt
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch.optim import Adamax


def create_objectworld_env(
    gamma: float = 1.0,
    grid_size: int = 10,
    T: int = 50,
    random_start: bool = False,
    continuous: bool = False,
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
        "n_objects": int(2 * grid_size),
    }

    env = ObjectWorldEnvironment(config_env)

    return env


def create_config_learner():
    config_default_learner = {
        "tol_exp": 0.005,
        "tol_var": 0.5,
        "miniter": 1,
        "maxiter": 5000,
    }

    return config_default_learner


if __name__ == "__main__":

    show = False
    store = True
    n_trajectories = 100
    experiment_name = "object-world"
    T = 20
    grid_size = 6
    gamma = 1.0
    random_start = False
    continuous = False
    config_default_learner = create_config_learner()
    learning_rate = {
        "scheduler": LambdaLR,
        "scheduler_kwargs": {
            "lr_lambda": lambda step: max(0.95 ** np.log(step + 1), 0.001)
        },
    }
    learning_rate_e = {
        "scheduler": ReduceLROnPlateau,
        "scheduler_kwargs": {"min_lr": 0.0005},
    }
    learning_rate_v = {
        "scheduler": ReduceLROnPlateau,
        "scheduler_kwargs": {"min_lr": 0.0005},
    }
    optimizer_e = Adamax
    optimizer_v = Adamax
    optimizer_e_kwargs = {"lr": 0.013481979369633936}
    optimizer_v_kwargs = {"lr": 0.06503275998396742, "eps": 1e-7, "weight_decay": 0.005}
    alternate_every = None
    var_factor = 5

    wandb.init(
        project=f"mceirl-{experiment_name}",
        name=f"{experiment_name}-g{grid_size}"
        + ("-random" if random_start else "")
        + ("-cont_features" if continuous else "-discrete_features"),
        config={
            "T": T,
            "grid_size": grid_size,
            "random_start": random_start,
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
    )
    config_default_learner = create_config_learner()

    fig = plt.figure()
    plt.pcolor(env.reward.reshape(env.grid_size, env.grid_size))
    plt.colorbar()
    plt.title("True reward")
    if show:
        plt.show()
    if store:
        plt.savefig(
            Path("plots") / "object_world_environment" / "real_rewards.jpg",
            format="jpg",
        )

    # create demonstrator
    demo = demonstrator.ObjectWorldDemonstrator(
        env,
        demonstrator_name="ObjectWorldDemonstrator",
        T=T,
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
        agent_name="AgentExpectation",
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
    wandb.log(
        {
            "reward_expectation": reward_expectation,
            "reward_diff_expectation": np.abs(reward_demonstrator - reward_expectation),
            "iterations_expectation": iter_expectation,
            "time_total_expectation": sum(time_expectation),
            "time_avg_per_iter_expectation": np.mean(time_expectation),
        }
    )

    # agent that also matches variances
    agent_variance = learner.TabularLearner(
        env,
        demo.mu_demonstrator,
        config_default_learner,
        agent_name="AgentVariance",
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

    wandb.log(
        {
            "reward_variance": reward_variance,
            "reward_diff_variance": np.abs(reward_demonstrator - reward_variance),
            "iterations_variance": iter_variance,
            "time_total_variance": sum(time_variance),
            "time_avg_per_iter_variance": np.mean(time_variance),
        }
    )

    wandb.finish()
