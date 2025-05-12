import agents.learner as learner
import agents.demonstrator as demonstrator
from environments.car_racing_environment import CarRacingEnvironment
from MDP_solver_approximation import MDPSolverApproximation
import numpy as np
import pandas as pd




def create_carracing_env(lap_complete_percent:float=0.95):

    config_env = {
        "gamma": 1.0,
        "lap_complete_percent": lap_complete_percent,
        "n_frames" : 4,
        "width": 64,
        "height": 64,
        "n_colors":6,
    }

    env = CarRacingEnvironment(config_env)
    return env


def create_config_learner():
    config_default_learner = {"tol": 0.0005, "miniter": 1, "maxiter": 400}

    return config_default_learner

if __name__ == "__main__":

    show = False
    store = True
    verbose = False

    
    n_trajectories = 100
    T = 1000

    # create the environment
    env = create_carracing_env()

    # Learner config
    config_default_learner = create_config_learner()

    # create demonstrator
    demo = demonstrator.CarRacingDemonstrator(
        env,
        demonstrator_name="CarRacingDemonstrator",
        T=T,
        n_trajectories=n_trajectories,
        solver = MDPSolverApproximation(T,True)
    )
    

    print("Demonstrator's expected value: ", demo.mu_demonstrator[0])
    print("Demonstrator's variance: ", demo.mu_demonstrator[1])

    reward_demonstrator = env.compute_true_reward_for_agent(demo, n_trajectories, T)

    if verbose:
        print("Demonstrator done")

    print("-- Results --")

    print("----- Demonstrator -----")
    print("reward: ", reward_demonstrator)

    demo.draw(show, store, 0)