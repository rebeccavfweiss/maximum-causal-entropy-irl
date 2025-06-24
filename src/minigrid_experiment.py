import agents.learner as learner
import agents.demonstrator as demonstrator
from environments.minigrid_environment import CrossingMiniGridEnvironment
import MDP_solver_exact as MDPSolver
import numpy as np
import pandas as pd
from random import randint
import wandb
from pathlib import Path


def create_minigrid_env(grid_size: int = 9):

    config_env = {
        "theta": np.diag([-1.0, -1.0, 0.5, 0.5, 10.0, -10.0]),
        "gamma": 1.0,
        "env_name": "MiniGrid-LavaCrossingS9N1-v0",
        "render_mode": "rgb_array",
        "grid_size": grid_size,
        "seed": randint(1, 100),
    }

    env = CrossingMiniGridEnvironment(config_env)
    return env


def create_config_learner():
    config_default_learner = {"tol": 0.005, "miniter": 1, "maxiter": 3000}

    return config_default_learner


if __name__ == "__main__":

    show = False
    store = False
    experiment_name = "minigrid"

    grid_sizes = [2 * i + 1 for i in range(2, 11, 2)]
    horizons = [2 * s + 2 for s in grid_sizes]
    runs = 3
    n_trajectories = None

    results = []

    for grid_size in grid_sizes:
        for T in horizons:
            if T < 2 * grid_size + 1:
                # agent has no chance to complete goal in the given horizon
                continue
            for i in range(runs):

                wandb.init(
                    project="mceirl-car-racing",
                    name=f"test-run",
                    config={
                        "grid_size": grid_size,
                        "horizon": T,
                        "n_trajectories": n_trajectories,
                        "run": i,
                    })

                # create the environment
                env = create_minigrid_env(grid_size)

                # Learner config
                config_default_learner = create_config_learner()

                # create demonstrator
                demo = demonstrator.CrossingMinigridDemonstrator(
                    env,
                    demonstrator_name="GymDemonstrator",
                    T=T,
                    n_trajectories=n_trajectories
                )
                demo.render(show, store, 0)

                reward_demonstrator = env.compute_true_reward_for_agent(demo, n_trajectories, T)

                wandb.log({"demonstrator_expected_value" : demo.mu_demonstrator[0],
                    "demonstrator_variance": demo.mu_demonstrator[1],
                    "demonstrator_reward": reward_demonstrator})

                # create agent that uses only expectation matching
                agent_expectation = learner.TabularLearner(
                    env,
                    demo.mu_demonstrator,
                    config_default_learner,
                    agent_name="AgentExpectation",
                    solver=MDPSolver.MDPSolverExactExpectation(T),
                )
                iter_expectation, time_expectation = agent_expectation.batch_MCE()
                agent_expectation.compute_and_draw(show, store, 2)
                reward_expectation = env.compute_true_reward_for_agent(agent_expectation, n_trajectories, T)

                wandb.log({
                    "reward_expectation": reward_expectation,
                    "reward_diff_expectation": np.abs(reward_demonstrator - reward_expectation),
                    "iterations_expectation": iter_expectation,
                    "time_total_expectation": sum(time_expectation),
                    "time_avg_per_iter_expectation": np.mean(time_expectation),
                })

                # create agent that also matches variances
                agent_variance = learner.TabularLearner(
                    env,
                    demo.mu_demonstrator,
                    config_default_learner,
                    agent_name="AgentVariance",
                    solver=MDPSolver.MDPSolverExactVariance(T),
                )
                iter_variance, time_variance = agent_variance.batch_MCE()
                agent_variance.compute_and_draw(show, store, 4)
                reward_variance = env.compute_true_reward_for_agent(
                    agent_variance, n_trajectories, T
                )

                wandb.log({
                    "reward_variance": reward_variance,
                    "reward_diff_variance": np.abs(reward_demonstrator - reward_variance),
                    "iterations_variance": iter_variance,
                    "time_total_variance": sum(time_variance),
                    "time_avg_per_iter_variance": np.mean(time_variance),
                })

                wandb.finish()

                results.append(
                    [
                        T,
                        grid_size,
                        i,
                        reward_demonstrator,
                        reward_expectation,
                        iter_expectation,
                        sum(time_expectation),
                        np.mean(time_expectation),
                        np.std(time_expectation),
                        reward_variance,
                        iter_variance,
                        sum(time_variance),
                        np.mean(time_variance),
                        np.std(time_variance),
                    ]
                )

            results_df = pd.DataFrame(
                results,
                columns=[
                    "T",
                    "grid",
                    "run",
                    "reward_demo",
                    "reward_exp",
                    "iter_exp",
                    "time_exp",
                    "mean_time_exp",
                    "std_time_exp",
                    "reward_var",
                    "iter_var",
                    "time_var",
                    "mean_time_var",
                    "std_time_var",
                ],
            )

            results_df.to_csv(Path("experiments") /"results_new.csv")
