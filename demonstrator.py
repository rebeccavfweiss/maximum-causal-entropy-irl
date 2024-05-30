import MDPSolver
from env_objectworld import Environment
import copy
import warnings

class Demonstrator:
    """
    class to implement the demonstrator that computes its policy for a certain reward using value iteration

    Parameters
    ----------
    env : env_objectworld.Environment
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
        self.mu_demonstrator = self.get_mu_usingRewardFeatures(self.env, self.env.reward)

    def get_mu_usingRewardFeatures(self, env:Environment, reward):
        """
        computes feature expectation and variance terms for the demonstrator using value iteration and computing the value function based on it

        Parameters
        ----------
        env : env_objectworld.Environment
            the environment representing the setting of the problem
        reward : ndarray
            reward for each state

        Returns
        -------
        ???????
        """
        Q, V, pi_d, pi_s = MDPSolver.MDPSolver.valueIteration(env, dict(reward=reward))
        self.V = V
        self.pi = pi_s
        _, mu, nu = MDPSolver.MDPSolver.computeFeatureSVF_bellmann_averaged(env, pi_s)
        return mu[:env.n_features_reward], nu[:env.n_features_reward,:env.n_features_reward]


    def draw(self):
        """
        draws the policy of the demonstrator as long as it has been computed before, else a warning is thrown
        """
        if self.pi is None:
            warnings.warn("Policy has not been computed yet.")
        else:
            self.reward = copy.deepcopy(self.env.reward)
            self.env.draw(self.V, self.pi, self.reward, False, self.demonstrator_name, 0)