"""
Generates the numbers for Table 1 in the paper.
"""

import agent
import demonstrator
import env_objectworld
import MDPSolver2
import numpy as np
import argparse
from tqdm import tqdm

n_average = 10 # number of object-worlds to average over

def create_env(rng=None):
    ####### create environment
    env = None

    config_env = {"gridsizefull": 10,
                "theta_e": [1, 0.9, 0.2],
                "theta_v": [[0.0,0.0,0.0],
                            [0.0,0.0,0.0],
                            [0.0,0.0,0.0]],
                "gamma": 0.99,
                "randomMoveProb": 0.0,
                "terminalState": 1,
                "terminal_gamma": 0.9,
                }
    env = env_objectworld.Environment(config_env, rng)

    return env

def create_config_learner():
    config_default_learner = {"tol": 0.0005,
                                "miniter": 25,
                                "maxiter": 300
                            }

    return config_default_learner

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description='Train an agent with preferences.')
    #parser.add_argument('Cr', type=float, help='soft reward constraints')
    #args = parser.parse_args()

    results_expectation = []
    results_variation = []
    for seed in tqdm(range(n_average)):
        rng = np.random.RandomState(seed)

        ####### create the environment
        env = create_env(rng)

        ####### Learner config
        config_default_learner = create_config_learner()

        ####### create teacher
        tAg = demonstrator.Demonstrator(env, myname="demonstrator")

        ####### create learner with preferences (with Agnostic teacher)
        learner = agent.Agent(env, tAg.mu_demonstrator, config_default_learner, myname="agent", solver=MDPSolver2.MDPSolverExpectation())
        learner.batch_MCE(verbose=False)
        learner.compute_and_draw(fignum=0, paper=False)
        reward = np.dot(env.reward, learner.solver.computeFeatureSVF_bellmann(env, learner.pi)[0])
        results_expectation.append(reward)

        learner = agent.Agent(env, tAg.mu_demonstrator, config_default_learner, myname="agent", solver=MDPSolver2.MDPSolverVariance())
        learner.batch_MCE(verbose=False)
        learner.compute_and_draw(fignum=0, paper=False)
        reward = np.dot(env.reward, learner.solver.computeFeatureSVF_bellmann(env, learner.pi)[0])
        results_variation.append(reward)



        tqdm.write("Result for seed %i: %f / %f" % (seed, results_expectation[-1], results_variation[-1]))

    
    results_expectation = np.array(results_expectation)
    results_variaation = np.array(results_variation)

    print("-- STATISTICS --")
    print("----Expectation -----")
    print("means:", np.mean(results_expectation))
    print("std-err:", np.std(results_expectation))
    print("")

    print("----Variation -----")
    print("means:", np.mean(results_variation))
    print("std-err:", np.std(results_variation))




