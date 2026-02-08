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


def create_objectworld_env(T: int = 50):

    config_env = {
        "theta": [
            1.0,
            1.0,
            -2.0,
        ],  # just any thing because the base class needs this variable, not used here
        "gamma": 1.0,
        "grid_size": 10,
        "random_start": False,
        "continuous": False,
        "T": T,
    }

    env = ObjectWorldEnvironment(config_env)

    return env


def create_config_learner():
    config_default_learner = {
        "tol_exp": 0.0005,
        "tol_var": 0.0025,
        "miniter": 1,
        "maxiter": 5000,
    }

    return config_default_learner


if __name__ == "__main__":

    show = False
    store = True
    n_trajectories = 50
    experiment_name = "object-world"
    T = 50

    wandb.init(
        project=f"mceirl-{experiment_name}",
        name=f"{experiment_name}-run_0",
        config={"T": T},
    )

    env = create_objectworld_env(T=T)
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

    demo.render(show, store, 1)
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
    )
    iter_variance, time_variance = agent_variance.batch_MCE()
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
