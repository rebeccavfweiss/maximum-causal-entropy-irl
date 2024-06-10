import agent
import demonstrator
import environment
import MDPSolver
import numpy as np

# number of object-worlds to average over
n_average = 1

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
    
    config_env2 = {"gridsizefull": 10,
                "theta_e": [1, 0.9, 0.2],
                "theta_v": [[0.0,0.0,0.0],
                            [0.0,0.0,0.0],
                            [0.0,0.0,0.0]],
                "gamma": 1.0,
                "randomMoveProb": 0.0,
                "terminalState": 0,
                "terminal_gamma": 1.0,
                }
    
    config_env3 = {"gridsizefull": 6,
                "theta_e": [0.5, 0.25, 1.75, -1.0],
                "theta_v": [[0.0,0.0,0.0, 0.0],
                            [0.0,0.0,0.0, 0.0],
                            [0.0,0.0,0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0]],
                "gamma": 1.0,
                "object_rewards": [0.5, 0.25, 1.75]}
    
    config_env4 = {"gridsizefull": 6,
                "theta_e": [1.0, -1.0],
                "theta_v": [[0.0, 0.0],
                            [0.0, 0.0]],
                "gamma": 0.5,
                "object_rewards": [10240/1023, 1566720/160177 , 497509888/5285841]}
    
    config_env5 = {"gridsizefull": 6,
                "theta_e": [1.0, -2.0],
                "theta_v": [[0.0, 0.0],
                            [0.0, 0.0]],
                "gamma": 1.0,
                "object_rewards": [0.5, 0.25, 1.75]}
    env = environment.Environment(config_env5, rng)
    env.reward = env.get_demonstrators_reward()

    return env

def create_config_learner():
    config_default_learner = {"tol": 0.0005,
                                "miniter": 10,
                                "maxiter": 300
                            }

    return config_default_learner

if __name__ == "__main__":

    show = True
    verbose = False
    T = 10

    rng = np.random.RandomState(0)

    # create the environment
    env = create_env(rng)

    # Learner config
    config_default_learner = create_config_learner()

    #print(env.reward.reshape((env.grid_size_full, env.grid_size_full)))
    #print(env.get_transition_matrix(randomMoveProb=0.0))

    # create teacher
    demo = demonstrator.Demonstrator(env, demonstrator_name="demonstrator")
    demo.draw(show)

    print("Demonstrator done")

    print(demo.mu_demonstrator)
    reward = np.dot(env.reward, demo.solver.computeFeatureSVF_bellmann(env, demo.pi)[0])
    print(reward)

    # create agent that uses only expectation matching
    learner = agent.Agent(env, demo.mu_demonstrator, config_default_learner, agent_name="agent_expectation", solver=MDPSolver.MDPSolverExpectation(T))
    learner.batch_MCE(verbose=verbose)

    print(learner.theta_e, learner.theta_v)
    learner.compute_and_draw(show)
    reward = np.dot(env.reward, learner.solver.computeFeatureSVF_bellmann(env, learner.pi)[0])
    results_expectation = reward

    print("first learner done")

    # create agent that also matches variances
    learner = agent.Agent(env, demo.mu_demonstrator, config_default_learner, agent_name="agent_variance", solver=MDPSolver.MDPSolverVariance(T))
    learner.batch_MCE(verbose=verbose)
    print(learner.theta_e, learner.theta_v)
    learner.compute_and_draw(show)
    reward = np.dot(env.reward, learner.solver.computeFeatureSVF_bellmann(env, learner.pi)[0])
    results_variance = reward

    print("second done")

    #results_expectation = np.array(results_expectation)
    #results_variance = np.array(results_variance)

    print("-- STATISTICS --")
    print("----Expectation -----")
    print("reward: ", results_expectation)
    #print("std-err:", np.std(results_expectation))
    print("")

    print("----variance -----")
    print("reward: ", results_variance)
    #print("std-err:", np.std(results_variance))




