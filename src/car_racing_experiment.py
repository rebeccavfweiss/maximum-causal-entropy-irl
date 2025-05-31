import agents.learner as learner
import agents.demonstrator as demonstrator
from environments.car_racing_environment import CarRacingEnvironment
from MDP_solver_approximation import MDPSolverApproximationExpectation, MDPSolverApproximationVariance
import numpy as np
import psutil
from utils import create_logger, log
import gc


def create_carracing_env(lap_complete_percent:float=0.95, T:int = 1000):

    config_env = {
        "gamma": 1.0,
        "lap_complete_percent": lap_complete_percent,
        "T":T,
        "n_frames" : 4,
        "width": 84,
        "height": 84,
    }

    env = CarRacingEnvironment(config_env)
    return env


def create_config_learner(n_trajectories:int = 1, maxiter:int=3):
    config_default_learner = {"tol": 0.0005, "miniter": 1, "maxiter": maxiter, "n_trajectories":n_trajectories}

    return config_default_learner

def log_memory(stage="", verbose:bool=False):
    mem = psutil.virtual_memory()
    log(f"[MEMORY] {stage}: {mem.available / (1024 ** 2):.2f} MB free", verbose)

if __name__ == "__main__":

    show = False
    store = True
    verbose = False

    maxiter = 30
    n_trajectories = 100
    sac_timesteps = 7500
    sac_buffer_size = 50000
    # does not really change anything so for now just limit T (i.e. technically goal of the agents now to just survive on the track as long as possible until time runs out as will not be possible to achieve lap in restricted time)
    lap_percent_complete=0.95
    T = 200

    create_logger("medium_experiment_log")

    log_memory("Start", verbose)

    # create the environment
    env = create_carracing_env(lap_complete_percent=lap_percent_complete, T=T)

    # Learner config
    config_default_learner = create_config_learner(n_trajectories, maxiter)

    log_memory("Environment + Config created", verbose)

    # create demonstrator
    demo = demonstrator.CarRacingDemonstrator(
        env,
        demonstrator_name="CarRacingDemonstrator",
        T=T,
        n_trajectories=n_trajectories,
        solver = MDPSolverApproximationExpectation(T,True, sac_timesteps, sac_buffer_size)
    )

    log_memory("Demonstrator created", verbose)

    log("Demonstrator's expected value: {}".format(demo.mu_demonstrator[0]), verbose)
    log("Demonstrator's variance: {}".format(demo.mu_demonstrator[1]), verbose)


    reward_demonstrator = env.compute_true_reward_for_agent(demo, n_trajectories, T)

    demo.draw(show, store, 0)

    # clean up
    del demo.policy
    log_memory("After demonstrator run + cleanup", verbose)

    log("Demonstrator done", verbose)

    # create agent that uses only expectation matching
    agent_expectation = learner.ApproximateLearner(
        env,
        demo.mu_demonstrator,
        config_default_learner,
        agent_name="AgentExpectation",
        solver=MDPSolverApproximationExpectation(T, sac_timesteps=sac_timesteps, sac_buffer_size=sac_buffer_size),
    )

    iter_expectation, time_expectation = agent_expectation.batch_MCE(verbose=verbose)
    agent_expectation.compute_and_draw(show, store, 2)
    reward_expectation = env.compute_true_reward_for_agent(agent_expectation, n_trajectories, T)

    log_memory("After expectation-only agent", verbose)    

    log("First agent done", verbose)

    log(f"""----- Expectation -----
    reward: {reward_expectation} (diff. to demonstrator: {np.abs(reward_demonstrator - reward_expectation)})
    theta_e: {agent_expectation.theta_e if verbose else 'hidden'}
    iterations used: {iter_expectation}
    time used (total/ avg. per iteration): {sum(time_expectation)} / {np.mean(time_expectation)}
    """, verbose)

    del agent_expectation
    gc.collect()
    log_memory("After expectation policy clean up", verbose)

    # create agent that also matches variances
    agent_variance = learner.ApproximateLearner(
        env,
        demo.mu_demonstrator,
        config_default_learner,
        agent_name="AgentVariance",
        solver=MDPSolverApproximationVariance(T, sac_timesteps=sac_timesteps, sac_buffer_size=sac_buffer_size),
    )

    iter_variance, time_variance = agent_variance.batch_MCE(verbose=verbose)
    agent_variance.compute_and_draw(show, store, 4)
    reward_variance = env.compute_true_reward_for_agent(agent_variance, n_trajectories, T)

    log_memory("After expectation + variance agent", verbose)

    # Free up memory
    del agent_variance.policy
    log_memory("After variance policy clean up", verbose)

    if verbose:
        log("Second agent done", verbose)

    log("----- Expectation + Variance -----", verbose)
    log(
        f"reward: {reward_variance} "
        f"(diff. to demonstrator: {np.abs(reward_demonstrator - reward_variance)})",
        verbose
    )

    log(f"theta_e: {agent_variance.theta_e}", verbose)
    log(f"theta_v: {agent_variance.theta_v}", verbose)

    log(f"iterations used: {iter_variance}", verbose)
    log(
        f"time used (total/ avg. per iteration): "
        f"{sum(time_variance)} / {np.mean(time_variance)}",
        verbose
    )