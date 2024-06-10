import MDPSolver
from environment import Environment
import copy


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
        finite horizon value for the MDP solver (default = 10)

    """

    def __init__(self, env: Environment, demonstrator_name: str, T: int = 10):
        self.V = None
        self.pi = None
        self.reward = None
        self.env = env
        self.demonstrator_name = demonstrator_name
        self.solver = MDPSolver.MDPSolverExpectation(T)
        self.mu_demonstrator = self.get_mu_usingRewardFeatures(
            self.env, self.env.reward
        )

    def get_mu_usingRewardFeatures(self, env: Environment, reward):
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
        _, V, _, pi_s = self.solver.valueIteration(env, dict(reward=reward))
        self.V = V
        self.pi = pi_s

        _, mu, nu = self.solver.computeFeatureSVF_bellmann_averaged(env, pi_s)

        return (
            mu[: env.n_features_reward],
            nu[: env.n_features_reward, : env.n_features_reward],
        )

    def draw(self, show=False):
        """
        draws the policy of the demonstrator as long as it has been computed before, else a warning is thrown
        """

        self.reward = copy.deepcopy(self.env.reward)
        self.env.draw(self.V, self.pi, self.reward, show, self.demonstrator_name, 0)
