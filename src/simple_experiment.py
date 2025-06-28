import agents.learner as learner
import agents.demonstrator as demonstrator
from environments.simple_environment import SimpleEnvironment
import MDP_solver_exact as MDPSolver
import numpy as np
import wandb


def create_simple_env():

    config_env = {
        "theta": [1.0, 1.0, -2.0],
        "gamma": 1.0,
    }

    env = SimpleEnvironment(config_env)

    return env


def create_config_learner():
    config_default_learner = {"tol": 0.0005, "miniter": 1, "maxiter": 3000}

    return config_default_learner


if __name__ == "__main__":

    show = False
    store = False
    experiment_name = "simple"

    T = 20
    n_trajectories = None

    wandb.init(
        project="mceirl-simple",
        name=f"{experiment_name}-run",
        config={
            "n_trajectories": n_trajectories,
            "T": T
        }
    )

    # create the environment
    env = create_simple_env()

    # Learner config
    config_default_learner = create_config_learner()

    # create demonstrator
    demo = demonstrator.SimpleDemonstrator(env, demonstrator_name="SimpleDemonstrator", T=T, n_trajectories=n_trajectories)

    demo.render(show, store, 0)

    reward_demonstrator = env.compute_true_reward_for_agent(demo, n_trajectories, T)


    wandb.log({"demonstrator_expected_value" : demo.mu_demonstrator[0],
        "demonstrator_variance": demo.mu_demonstrator[1],
        "demonstrator_reward": reward_demonstrator,
        "theta_*": env.theta_reward})

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
    reward_expectation = env.compute_true_reward_for_agent(
        agent_expectation, n_trajectories, T
    )
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
