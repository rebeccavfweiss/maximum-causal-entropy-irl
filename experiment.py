import agent
import demonstrator
import environment
import MDPSolver
import numpy as np


def create_env():
    
    config_env = {"gridsizefull": 6,
                "theta_e": [1.0, -1.0],
                "theta_v": [[0.0, 0.0],
                            [0.0, 0.0]],
                "gamma": 1.0,
                "object_rewards": [0.5, 0.25, 1.75]}
    
    env = environment.Environment(config_env)

    return env

def create_config_learner():
    config_default_learner = {"tol": 0.0005,
                                "miniter": 10,
                                "maxiter": 300
                            }

    return config_default_learner

if __name__ == "__main__":

    show = False
    store = True
    verbose = False

    # create the environment
    env = create_env()

    # Learner config
    config_default_learner = create_config_learner()

    # create demonstrator
    demo = demonstrator.Demonstrator(env, demonstrator_name="Demonstrator")
    demo.draw(show, store, 0)

    if verbose:
        print("Demonstrator done")

    reward_demonstrator = np.dot(env.reward, demo.solver.computeFeatureSVF_bellmann(env, demo.pi)[0])

    # create agent that uses only expectation matching
    agent_expectation = agent.Agent(env, demo.mu_demonstrator, config_default_learner, agent_name="Agent Expectation", solver=MDPSolver.MDPSolverExpectation())
    agent_expectation.batch_MCE(verbose=verbose)
    agent_expectation.compute_and_draw(show, store, 2)
    reward_expectation = np.dot(env.reward, agent_expectation.solver.computeFeatureSVF_bellmann(env, agent_expectation.pi)[0])

    if verbose:
        print("First agent done")

    # create agent that also matches variances
    agent_variance = agent.Agent(env, demo.mu_demonstrator, config_default_learner, agent_name="Agent Variance", solver=MDPSolver.MDPSolverVariance())
    agent_variance.batch_MCE(verbose=verbose)
    agent_variance.compute_and_draw(show, store, 4)
    reward_variance = np.dot(env.reward, agent_variance.solver.computeFeatureSVF_bellmann(env, agent_variance.pi)[0])

    if verbose:
        print("Second agent done")

    print("-- Results --")

    print("----- Demonstrator -----")
    print("reward: ", reward_demonstrator)
    print("theta_*: ", env.theta_e)
    print("")

    print("----- Expectation -----")
    print("reward: ", reward_expectation, " (diff. to demonstrator: ", np.abs(reward_demonstrator-reward_expectation), ")")
    print("theta_e: ", agent_expectation.theta_e)
    print("policy difference:", sum(np.linalg.norm(demo.pi[i,:,:] - agent_expectation.pi[i,:,:], ord = "fro") for i in range(demo.pi.shape[0])))
    print("")

    print("----- Expectation + Variance -----")
    print("reward: ", reward_variance, " (diff. to demonstrator: ", np.abs(reward_demonstrator-reward_variance), ")")
    print("theta_e: ", agent_variance.theta_e)
    print("theta_v: ", agent_variance.theta_v)
    print("policy difference:", sum(np.linalg.norm(demo.pi[i,:,:] - agent_variance.pi[i,:,:], ord ="fro") for i in range(demo.pi.shape[0])))
    