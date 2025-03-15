import MDPSolver
from environment import Environment
import copy
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
    T : int
        finite horizon value for the MDP solver (default = 45)
    """

    def __init__(self, env: Environment, demonstrator_name: str, T: int = 45):
        self.V = None
        self.pi = None
        self.reward = None
        self.env = env
        self.demonstrator_name = demonstrator_name
        self.T = T
        self.solver = MDPSolver.MDPSolverExpectation(T, compute_variance=True)
        self.mu_demonstrator = self.get_mu_usingRewardFeatures()

    def get_mu_using_reward_features(self):
        """
        computes feature expectation and variance terms for the demonstrator using a predefined policy and computing the value function based on it

        Returns
        -------
        feature expectation and variance arrays
        """

        pi_s = np.zeros((self.T, self.env.n_states, self.env.n_actions))

        # define the demonstrator's behavior in a specific way
        for t in range(pi_s.shape[0]):
            pi_s[t, self.env.point_to_int(4, 0), self.env.actions["up"]] = 0.5
            pi_s[t, self.env.point_to_int(4, 0), self.env.actions["down"]] = 0.5

            pi_s[t, self.env.point_to_int(3, 0), self.env.actions["up"]] = 1.0
            pi_s[t, self.env.point_to_int(2, 0), self.env.actions["up"]] = 1.0
            pi_s[t, self.env.point_to_int(1, 0), self.env.actions["up"]] = 1.0

            pi_s[t, self.env.point_to_int(5, 0), self.env.actions["down"]] = 1.0
            pi_s[t, self.env.point_to_int(6, 0), self.env.actions["down"]] = 1.0
            pi_s[t, self.env.point_to_int(7, 0), self.env.actions["down"]] = 1.0

        self.V = self.compute_value_function(pi_s)
        self.pi = pi_s

        _, mu, nu = self.solver.compute_feature_SVF_bellmann_averaged(self.env, pi_s)

        return (
            mu,
            nu,
        )

    def compute_value_function(self, pi):
        """
        Computes the state-dependent value function based on the given policy

        Parameters
        ----------
        pi : ndarray
            policy to use

        Returns
        -------
        ndarray
            matrix containing the time-dependent value function
        """
        V = np.zeros((self.T + 1, self.env.n_states))

        for t in range(self.T - 1, -1, -1):
            for s in range(self.env.n_states):
                V[t, s] = sum(
                    pi[t, s, a]
                    * sum(
                        self.env.T[s, s_prime, a]
                        * (self.env.reward[s] + self.env.gamma * V[t + 1, s_prime])
                        for s_prime in range(self.env.n_states)
                    )
                    for a in range(self.env.n_actions)
                )

        return V

    def draw(self, show: bool = False, store: bool = False, fignum: int = 0):
        """
        draws the policy of the demonstrator as long as it has been computed before, else a warning is thrown

        Parameters
        ----------
        show : bool
            whether or not the plot should be shown (default = False)
        store : bool
            whether or not the plot should be stored (default = False)
        fignum : int
            identifier number for the figure (default = 0)
        """

        self.reward = copy.deepcopy(self.env.reward)
        self.env.draw(
            self.V, self.pi, self.reward, show, self.demonstrator_name, fignum, store
        )
