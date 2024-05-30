import agent
import demonstrator
import env_objectworld
import MDPSolver
import numpy as np
from tqdm import tqdm

# number of object-worlds to average over
n_average = 10 

def create_env(rng=None):
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

    results_expectation = []
    results_variance = []

    for seed in tqdm(range(n_average)):
        rng = np.random.RandomState(seed)

        # create the environment
        env = create_env(rng)

        # Learner config
        config_default_learner = create_config_learner()

        # create teacher
        tAg = demonstrator.Demonstrator(env, demonstrator_name="demonstrator")

        # create agent that uses only expectation matching
        learner = agent.Agent(env, tAg.mu_demonstrator, config_default_learner, agent_name="agent_expectation", solver=MDPSolver.MDPSolverExpectation())
        learner.batch_MCE(verbose=False)
        learner.compute_and_draw()
        reward = np.dot(env.reward, learner.solver.computeFeatureSVF_bellmann(env, learner.pi)[0])
        results_expectation.append(reward)

        # create agent that also matches variances
        learner = agent.Agent(env, tAg.mu_demonstrator, config_default_learner, agent_name="agent_variance", solver=MDPSolver.MDPSolverVariance())
        learner.batch_MCE(verbose=False)
        learner.compute_and_draw()
        reward = np.dot(env.reward, learner.solver.computeFeatureSVF_bellmann(env, learner.pi)[0])
        results_variance.append(reward)

        tqdm.write("Result for seed %i: %f / %f" % (seed, results_expectation[-1], results_variance[-1]))

    
    results_expectation = np.array(results_expectation)
    results_variaation = np.array(results_variance)

    print("-- STATISTICS --")
    print("----Expectation -----")
    print("means:", np.mean(results_expectation))
    print("std-err:", np.std(results_expectation))
    print("")

    print("----variance -----")
    print("means:", np.mean(results_variance))
    print("std-err:", np.std(results_variance))




