import agents.learner as learner
import agents.demonstrator as demonstrator
from environments.minigrid_environment import CrossingMiniGridEnvironment
import MDP_solver_exact as MDPSolver
import numpy as np
import pandas as pd
from random import randint
import wandb
from multiprocessing import Pool
from itertools import product
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

def run_experiment(args):
    grid_size, T, i = args

    run_name = f"minigrid_g{grid_size}_T{T}_run{i}"
    wandb.init(
        project="mceirl-minigrid",
        name=run_name,
        config={
            "grid_size": grid_size,
            "horizon": T,
            "run": i,
        },
        reinit="finish_previous"
    )

    print(f"✅ W&B run started: {run_name}")

    # Create the environment, learner, demonstrator, etc
    env = create_minigrid_env(grid_size)
    config_default_learner = create_config_learner()

    demo = demonstrator.CrossingMinigridDemonstrator(
        env,
        demonstrator_name="MiniGridDemonstrator",
        T=T,
    )
    demo.draw(False, False, 0)
    reward_demonstrator = env.compute_true_reward_for_agent(demo, None, T)

    wandb.log({
        "demonstrator_expected_value": demo.mu_demonstrator[0],
        "demonstrator_variance": demo.mu_demonstrator[1],
        "demonstrator_reward": reward_demonstrator,
    })

    # Expectation matching agent
    agent_expectation = learner.TabularLearner(
        env,
        demo.mu_demonstrator,
        config_default_learner,
        agent_name="AgentExpectation",
        solver=MDPSolver.MDPSolverExactExpectation(T),
    )
    iter_expectation, time_expectation = agent_expectation.batch_MCE()
    reward_expectation = env.compute_true_reward_for_agent(agent_expectation, None, T)
    wandb.log({
        "reward_expectation": reward_expectation,
        "iterations_expectation": iter_expectation,
        "time_total_expectation": sum(time_expectation),
        "time_avg_per_iter_expectation": np.mean(time_expectation),
    })

    print(f"✅ Finished experiment with grid={grid_size}, T={T}, run={i}")

    # Variance matching agent
    agent_variance = learner.TabularLearner(
        env,
        demo.mu_demonstrator,
        config_default_learner,
        agent_name="AgentVariance",
        solver=MDPSolver.MDPSolverExactVariance(T),
    )
    iter_variance, time_variance = agent_variance.batch_MCE()
    reward_variance = env.compute_true_reward_for_agent(agent_variance, None, T)
    wandb.log({
        "reward_variance": reward_variance,
        "iterations_variance": iter_variance,
        "time_total_variance": sum(time_variance),
        "time_avg_per_iter_variance": np.mean(time_variance),
    })

    wandb.finish()

    return [
        T, grid_size, i,
        reward_demonstrator, reward_expectation, iter_expectation,
        sum(time_expectation), np.mean(time_expectation), np.std(time_expectation),
        reward_variance, iter_variance, sum(time_variance),
        np.mean(time_variance), np.std(time_variance)
    ]

if __name__ == "__main__":


    grid_sizes = [2 * i + 1 for i in range(2, 5, 2)]
    horizons = [2 * s + 2 for s in grid_sizes]
    runs = 1

    tasks = []
    for grid_size in grid_sizes:
        for T in horizons:
            if T < 2 * grid_size + 1:
                continue
            for i in range(runs):
                tasks.append((grid_size, T, i))

    with Pool() as pool:
        results = pool.map(run_experiment, tasks)

    results_df = pd.DataFrame(
        results,
        columns=[
            "T","grid","run",
            "reward_demo","reward_exp","iter_exp",
            "time_exp","mean_time_exp","std_time_exp",
            "reward_var","iter_var","time_var",
            "mean_time_var","std_time_var"
        ]
    )
    results_df.to_csv(Path("experiments") / "minigrid"/ "results_parallel.csv")

