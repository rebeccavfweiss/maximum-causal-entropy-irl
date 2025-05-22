import agents.learner as learner
import agents.demonstrator as demonstrator
from environments.simple_environment import SimpleEnvironment
import MDP_solver_exact as MDPSolver
import numpy as np


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
    verbose = False

    T = 20
    n_trajectories = None

    # create the environment
    env = create_simple_env()

    # Learner config
    config_default_learner = create_config_learner()

    # create demonstrator
    demo = demonstrator.SimpleDemonstrator(env, demonstrator_name="SimpleDemonstrator", T=T, n_trajectories=n_trajectories)

    demo.draw(show, store, 0)
    print("Demonstrator's expected value: ", demo.mu_demonstrator[0])
    print("Demonstrator's variance: ", demo.mu_demonstrator[1])

    if verbose:
        print("Demonstrator done")

    reward_demonstrator = env.compute_true_reward_for_agent(demo, n_trajectories, T)

    print("-- Results --")

    print("----- Demonstrator -----")
    print("reward: ", reward_demonstrator)
    if verbose:
        print("theta_*: ", env.theta_reward)
        print("")

    # create agent that uses only expectation matching
    agent_expectation = learner.TabularLearner(
        env,
        demo.mu_demonstrator,
        config_default_learner,
        agent_name="Agent Expectation",
        solver=MDPSolver.MDPSolverExactExpectation(T),
    )
    iter_expectation, time_expectation = agent_expectation.batch_MCE(verbose=verbose)
    agent_expectation.compute_and_draw(show, store, 2)
    reward_expectation = env.compute_true_reward_for_agent(
        agent_expectation, n_trajectories, T
    )

    if verbose:
        print("First agent done")


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
    agent_variance = learner.TabularLearner(
        env,
        demo.mu_demonstrator,
        config_default_learner,
        agent_name="Agent Variance",
        solver=MDPSolver.MDPSolverExactVariance(T),
    )
    iter_variance, time_variance = agent_variance.batch_MCE(verbose=verbose)
    agent_variance.compute_and_draw(show, store, 4)
    reward_variance = env.compute_true_reward_for_agent(
        agent_variance, n_trajectories, T
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
