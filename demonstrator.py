import MDPSolver
from environment import Environment
import copy
import warnings
import numpy as np

class Demonstrator:
    """
    class to implement the demonstrator that computes its policy for a certain reward using value iteration

    Parameters
    ----------
    env : environment.Environment
        the environment representing the setting of the problem
    demonstrator_name : str
        name of the demonstrator
    
    """
    def __init__(self, env:Environment, demonstrator_name:str):
        self.V = None
        self.pi = None
        self.reward = None
        self.env = env
        self.demonstrator_name = demonstrator_name
        self.solver = MDPSolver.MDPSolverExpectation(10)
        self.mu_demonstrator = self.get_mu_usingRewardFeatures(self.env, self.env.reward)
        

    def get_mu_usingRewardFeatures(self, env:Environment, reward):
        """
        computes feature expectation and variance terms for the demonstrator using value iteration and computing the value function based on it

        Parameters
        ----------
        env : environment.Environment
            the environment representing the setting of the problem
        reward : ndarray
            reward for each state

        Returns
        -------
        feature expectation and variance arrays restricted to the reward features
        """
        Q, V, pi_d, pi_s = self.solver.valueIteration(env, dict(reward=reward))
        self.V = V
        self.pi = pi_s

        #self.pi[0,0,3] += 0.1
        #self.pi[1,1,3] += 0.1
        #self.pi[2,2,3] += 0.1
        #self.pi[3,3,3] += 0.1
        #self.pi[4,4,3] += 0.1
        #self.pi[5,5,2] += 0.1
        #self.pi[6,11,2] += 0.1
        #self.pi[7,17,2] += 0.1
        #self.pi[8,23,2] += 0.1
        #self.pi[9,29,2] += 0.1
 
        #pi_s = self.pi

        _, mu, nu = self.solver.computeFeatureSVF_bellmann_averaged(env, pi_s)
        return mu[:env.n_features_reward], nu[:env.n_features_reward,:env.n_features_reward]


    def draw(self, show =False):
        """
        draws the policy of the demonstrator as long as it has been computed before, else a warning is thrown
        """
        if self.pi is None:
            warnings.warn("Policy has not been computed yet.")
        else:
            self.reward = copy.deepcopy(self.env.reward)
            self.env.draw(self.V, self.pi, self.reward, show, self.demonstrator_name, 0)