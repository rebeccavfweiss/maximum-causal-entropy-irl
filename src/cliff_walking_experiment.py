"""
CliffWalking experiment runner.

Runs TabularLearner (expectation + variance) across different success rates
to demonstrate that variance matching outperforms expectation matching in
stochastic environments where the safe (high-variance) path matters.

Usage:
    python -m cliff_walking_experiment
"""

import os
import agents.learner as learner
import agents.demonstrator as demonstrator
from environments.cliff_walking_environment import CliffWalkingEnvironment
import solvers.MDP_solver_exact as MDPSolver
import numpy as np
import pandas as pd
from multiprocessing import Pool
import wandb
from pathlib import Path


def create_cliff_walking_env(success_rate: float, one_hot_features: bool = True):
    # theta = reward per state (used directly regardless of feature type)
    n_states = 49  # 48 grid + 1 absorbing
    theta = np.zeros(n_states)

    # Step cost for all non-terminal grid states
    for s in range(48):
        theta[s] = -1.0

    # Cliff states: large negative reward (falling off is bad)
    for s in CliffWalkingEnvironment.CLIFF_STATES:
        theta[s] = -100.0

    # Goal state: large positive reward
    # theta[CliffWalkingEnvironment.GOAL_STATE] = 10.0

    # Start state: neutral
    theta[CliffWalkingEnvironment.START_STATE] = 0.0

    # Terminal absorbing state: 0
    theta[48] = 0.0

    config_env = {
        "theta": theta,
        "gamma": 1.0,
        "success_rate": success_rate,
        "T": 100,
        "one_hot_features": one_hot_features,
    }

    return CliffWalkingEnvironment(config_env)


def create_config_learner():
    return {
        "tol_exp": 0.001,
        "tol_var": 0.005,
        "miniter": 1,
        "maxiter": 3000,
    }


def run_experiment(args):
    success_rate, T, i, one_hot_features = args

    feat_tag = "onehot" if one_hot_features else "scalar"
    run_name = f"cliff_sr{success_rate}_T{T}_{feat_tag}_run{i}"
    wandb.init(
        project="mceirl-cliffwalking",
        name=run_name,
        config={
            "success_rate": success_rate,
            "horizon": T,
            "run": i,
            "one_hot_features": one_hot_features,
        },
        reinit="finish_previous",
    )

    env = create_cliff_walking_env(success_rate, one_hot_features)
    config_default_learner = create_config_learner()

    demo = demonstrator.CliffWalkingDemonstrator(
        env,
        demonstrator_name="CliffWalkingDemonstrator",
        T=T,
    )
    path_to_file = demo.render(False, True, 0)
    if path_to_file is not None:
        wandb.log(
            {
                "eval/video_demonstrator": wandb.Video(
                    str(path_to_file), fps=2, format="mp4"
                )
            }
        )
    reward_demonstrator = env.compute_true_reward_for_agent(demo, None, T)

    wandb.log(
        {
            "demonstrator_expected_value": demo.mu_demonstrator[0],
            "demonstrator_variance": demo.mu_demonstrator[1],
            "demonstrator_reward": reward_demonstrator,
            "success_rate": success_rate,
        }
    )

    # Expectation matching agent
    agent_expectation = learner.TabularLearner(
        env,
        demo.mu_demonstrator,
        config_default_learner,
        agent_name="AgentExpectation",
        solver=MDPSolver.MDPSolverExactExpectation(T),
    )
    iter_expectation, time_expectation = agent_expectation.batch_MCE()
    agent_expectation.compute_and_draw(False, True, 2)
    path_to_file = agent_expectation.render(False, True, 6)
    if path_to_file is not None:
        wandb.log(
            {
                "eval/video_expectation": wandb.Video(
                    str(path_to_file), fps=2, format="mp4"
                )
            }
        )

    reward_expectation = env.compute_true_reward_for_agent(agent_expectation, None, T)
    wandb.log(
        {
            "reward_expectation": reward_expectation,
            "reward_diff_expectation": reward_expectation - reward_demonstrator,
            "iterations_expectation": iter_expectation,
            "time_total_expectation": sum(time_expectation),
            "time_avg_per_iter_expectation": np.mean(time_expectation),
        }
    )

    # Variance matching agent
    agent_variance = learner.TabularLearner(
        env,
        demo.mu_demonstrator,
        config_default_learner,
        agent_name="AgentVariance",
        solver=MDPSolver.MDPSolverExactVariance(T),
    )
    iter_variance, time_variance = agent_variance.batch_MCE()
    agent_variance.compute_and_draw(False, True, 4)
    path_to_file = agent_variance.render(False, True, 8)
    if path_to_file is not None:
        wandb.log(
            {"eval/video_variance": wandb.Video(str(path_to_file), fps=2, format="mp4")}
        )

    reward_variance = env.compute_true_reward_for_agent(agent_variance, None, T)
    wandb.log(
        {
            "reward_variance": reward_variance,
            "reward_diff_variance": reward_variance - reward_demonstrator,
            "iterations_variance": iter_variance,
            "time_total_variance": sum(time_variance),
            "time_avg_per_iter_variance": np.mean(time_variance),
        }
    )

    wandb.finish()

    return [
        success_rate,
        T,
        i,
        one_hot_features,
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


if __name__ == "__main__":

    success_rates = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 1.0 / 3.0]
    T = 100
    runs = 5

    tasks = []
    for sr in success_rates:
        for one_hot in [True, False]:
            for i in range(runs):
                tasks.append((sr, T, i, one_hot))

    with Pool(processes=10) as pool:
        results = pool.map(run_experiment, tasks)

    results_df = pd.DataFrame(
        results,
        columns=[
            "success_rate",
            "T",
            "run",
            "one_hot_features",
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

    os.makedirs(Path("experiments") / "cliff_walking", exist_ok=True)
    results_df.to_csv(Path("experiments") / "cliff_walking" / "results_parallel.csv")
    print(results_df.groupby(["success_rate", "one_hot_features"]).mean())
