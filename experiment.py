import agent
import demonstrator
from simple_environment import SimpleEnvironment
from gymnasium_environment import MiniGridCrossingEnvironment
import MDPSolver
import numpy as np
from random import randint


def create_simple_env():

    config_env = {
        "theta": [1.0, 1.0, -2.0],
        "gamma": 1.0,
    }

    env = SimpleEnvironment(config_env)

    return env


def create_minigrid_env():

    config_env = {
        "theta": np.diag([-1.0, -1.0, 0.5, 0.5, 10.0, -10.0]),
        "gamma": 1.0,
        "env_name": "MiniGrid-LavaCrossingS9N1-v0",
        "render_mode": "rgb_array",
        "grid_size": 9,
        "seed": randint(1, 100),
    }

    env = MiniGridCrossingEnvironment(config_env)
    return env


def create_config_learner():
    config_default_learner = {"tol": 0.0005, "miniter": 1, "maxiter": 300}

    return config_default_learner


if __name__ == "__main__":

    show = False
    store = False
    verbose = False
    n_training_episodes = 5000
    T = 30

    # create the environment
    # env = create_simple_env()
    env = create_minigrid_env()

    # Learner config
    config_default_learner = create_config_learner()

    # create demonstrator
    # demo = demonstrator.SimpleDemonstrator(env, demonstrator_name="SimpleDemonstrator", T=T)
    demo = demonstrator.GymDemonstrator(
        env,
        demonstrator_name="GymDemonstrator",
        T=T,
        n_training_episodes=n_training_episodes,
    )
    demo.draw(show, store, 0)
    print("Demonstrator's expected value: ", demo.mu_demonstrator[0])
    print("Demonstrator's variance: ", demo.mu_demonstrator[1])

    if verbose:
        print("Demonstrator done")

    reward_demonstrator = np.dot(
        env.reward, demo.solver.compute_feature_SVF_bellmann(env, demo.pi)[0]
    )

    # create agent that uses only expectation matching
    agent_expectation = agent.Agent(
        env,
        demo.mu_demonstrator,
        config_default_learner,
        agent_name="Agent Expectation",
        solver=MDPSolver.MDPSolverExpectation(T),
    )
    iter_expectation, time_expectation = agent_expectation.batch_MCE(verbose=verbose)
    agent_expectation.compute_and_draw(show, store, 2)
    reward_expectation = np.dot(
        env.reward,
        agent_expectation.solver.compute_feature_SVF_bellmann(
            env, agent_expectation.pi
        )[0],
    )

    if verbose:
        print("First agent done")

    print("-- Results --")

    print("----- Demonstrator -----")
    print("reward: ", reward_demonstrator)
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
    agent_variance = agent.Agent(
        env,
        demo.mu_demonstrator,
        config_default_learner,
        agent_name="Agent Variance",
        solver=MDPSolver.MDPSolverVariance(T),
    )
    iter_variance, time_variance = agent_variance.batch_MCE(verbose=verbose)
    agent_variance.compute_and_draw(show, store, 4)
    reward_variance = np.dot(
        env.reward,
        agent_variance.solver.compute_feature_SVF_bellmann(env, agent_variance.pi)[0],
    )

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
    print("theta_e: ", agent_variance.theta_e)
    print("theta_v: ", agent_variance.theta_v)

    print("iterations used: ", iter_variance)
    print(
        "time used (total/ avg. per iteration): ",
        sum(time_variance),
        "/",
        np.mean(time_variance),
    )
