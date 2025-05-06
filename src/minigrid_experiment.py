import agents.learner as learner
import agents.demonstrator as demonstrator
from environments.minigrid_environment import CrossingMiniGridEnvironment
import MDPSolver
import numpy as np
import pandas as pd
from random import randint



def create_minigrid_env(grid_size:int = 9):

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
    config_default_learner = {"tol": 0.0005, "miniter": 1, "maxiter": 400}

    return config_default_learner


if __name__ == "__main__":

    show = False
    store = False
    verbose = False

    
    grid_sizes = [5] #[2*i + 1 for i in range(4,6)]
    horizons = [11] #[2*s + 2 for s in grid_sizes]
    runs = 1
    n_trajectories = 1

    results = []

    for grid_size in grid_sizes:
        for T in horizons:
            if (T < 2*grid_size+1):
                # agent has no chance to complete goal in the given horizon
                continue
            for i in range(runs):

                # create the environment
                env = create_minigrid_env(grid_size)

                print("T = ", T)
                print("grid_size = ", grid_size)
                print("run = ", i)


                # Learner config
                config_default_learner = create_config_learner()

                # create demonstrator
                demo = demonstrator.GymDemonstrator(
                    env,
                    demonstrator_name="GymDemonstrator",
                    T=T,
                )
                demo.draw(show, store, 0)
                print("Demonstrator's expected value: ", demo.mu_demonstrator[0])
                print("Demonstrator's variance: ", demo.mu_demonstrator[1])

                if verbose:
                    print("Demonstrator done")

                reward_demonstrator = env.compute_true_reward_for_agent(demo, n_trajectories, T)

                # create agent that uses only expectation matching
                agent_expectation = learner.Learner(
                    env,
                    demo.mu_demonstrator,
                    config_default_learner,
                    agent_name="Agent Expectation",
                    solver=MDPSolver.MDPSolverExpectation(T),
                )
                iter_expectation, time_expectation = agent_expectation.batch_MCE(verbose=verbose)
                agent_expectation.compute_and_draw(show, store, 2)
                reward_expectation = env.compute_true_reward_for_agent(agent_expectation, n_trajectories, T)

                if verbose:
                    print("First agent done")

                print("-- Results --")

                print("----- Demonstrator -----")
                print("reward: ", reward_demonstrator)
                if verbose:
                    print("theta_*: ", env.theta_reward)
                    print("")

                print("----- Expectation -----")
                print(
                    "reward: ",
                    reward_expectation,
                    " (diff. to demonstrator: ",
                    np.abs(reward_demonstrator - reward_expectation),
                    ")",
                )
                if verbose: 
                    print("theta_e: ", agent_expectation.theta_e)

                print("iterations used: ", iter_expectation)
                print(
                    "time used (total/ avg. per iteration): ",
                    sum(time_expectation),
                    "/",
                    np.mean(time_expectation),
                )
                print("")

                # create agent that also matches variances
                agent_variance = learner.Learner(
                    env,
                    demo.mu_demonstrator,
                    config_default_learner,
                    agent_name="Agent Variance",
                    solver=MDPSolver.MDPSolverVariance(T),
                )
                iter_variance, time_variance = agent_variance.batch_MCE(verbose=verbose)
                agent_variance.compute_and_draw(show, store, 4)
                reward_variance = env.compute_true_reward_for_agent(agent_variance, n_trajectories, T)

                if verbose:
                    print("Second agent done")

                print("----- Expectation + Variance -----")
                print(
                    "reward: ",
                    reward_variance,
                    " (diff. to demonstrator: ",
                    np.abs(reward_demonstrator - reward_variance),
                    ")",
                )
                if verbose:
                    print("theta_e: ", agent_variance.theta_e)
                    print("theta_v: ", agent_variance.theta_v)

                print("iterations used: ", iter_variance)
                print(
                    "time used (total/ avg. per iteration): ",
                    sum(time_variance),
                    "/",
                    np.mean(time_variance),
                )

                results.append([T, grid_size, i, reward_demonstrator, reward_expectation, iter_expectation, sum(time_expectation), np.mean(time_expectation),np.std(time_expectation),
                                reward_variance, iter_variance, sum(time_variance), np.mean(time_variance), np.std(time_variance)])

            results_df = pd.DataFrame(results, columns=["T", "grid", "run","reward_demo", "reward_exp", "iter_exp", "time_exp", "mean_time_exp", "std_time_exp",
                                                        "reward_var", "iter_var", "time_var", "mean_time_var", "std_time_var"])
            
            results_df.to_csv("experiments\\results_new.csv")
