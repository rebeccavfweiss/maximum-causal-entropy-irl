import MDPSolver
import numpy as np
import copy
from environment import Environment

_largenum = 1000000


class Agent:
    """
    Implementing an agent using either expectation matching / expectation and variance matching to solve an IRL problem.

    Parameters
    ----------
    env : environment.Environment
        the environment representing the setting of the problem
    mu_demonstrator : tuple[float, float]
        feature expectation and variance terms of the demonstrator
    config_agent : dict[str: any]
        different configuration parameters for the agent including
        tol : convergence tolerance for batch_MCE
        maxiter : maximal number of iterations for batch_MCE
        miniter : minimal number of iterations for batch_MCE
    agent_name : str
        name of the agent
    solver : MDPSolver.MDPSolver
        solver to use (either only expectation matching or also variance matching)
    """

    def __init__(
        self,
        env: Environment,
        mu_demonstrator: tuple[float, float],
        config_agent: dict[str:any],
        agent_name: str,
        solver: MDPSolver.MDPSolver,
    ):
        self.env = env
        self.mu_demonstrator = mu_demonstrator
        self.theta_e = np.zeros(self.env.n_features_reward)
        self.theta_v = np.zeros(
            (self.env.n_features_reward, self.env.n_features_reward)
        )

        self.tol = config_agent["tol"]
        self.maxiter = config_agent["maxiter"]
        self.miniter = config_agent["miniter"]

        self.theta_upperBound = _largenum

        self.V = None
        self.pi = None
        self.reward = None
        self.solver = solver

        self.agent_name = agent_name

    def compute_and_draw(self, show: bool = False, store:bool = False, fignum:int=0):
        """
        computes soft_value iteration for given thetas and policy based on the result and draws policy

        Parameters
        ----------
        show : bool
            whether or not the plots should be shown (default = false)
        store : bool
            whether or not the plots should be stored (default = false)
        fignum : int
            identifier number for figure (default = 0)
        """
        self.reward = self.get_reward_for_given_thetas()
        self.variance = self.get_variance_for_given_thetas()
        _, _, pi_agent = self.solver.soft_valueIteration(
            self.env, dict(reward=self.reward, variance=self.variance)
        )
        self.pi = pi_agent

        self.V = self.solver.computeValueFunction_bellmann_averaged(
            self.env, self.pi, dict(reward=self.env.reward, variance=self.env.variance)
        )  # this is value of agent's policy w.r.t. env's reward
        self.env.draw(self.V, self.pi, self.reward, show, self.agent_name, fignum, store)

    def get_reward_for_given_thetas(self):
        """
        computes the reward based on theta_e for every state

        Returns
        -------
        reward : numpy.ndarray
        """
        reward = self.theta_e
        reward = self.env.get_reward_for_given_theta(reward)
        return reward

    def get_variance_for_given_thetas(self):
        """
        computes the variance term for every state needed for soft value iteration based on theta_v

        Returns
        -------
        variance : numpy.ndarray
        """
        variance = self.theta_v
        variance = self.env.get_variance_for_given_theta(variance)
        return variance

    def get_mu_soft(self):
        """
        computes feature expectation and variance terms based on soft value iteration and computing the corresponding value function

        Returns
        -------
        feature expectation and variance, once restricted to the reward features, once the full arrays
        """
        reward_agent = self.get_reward_for_given_thetas()
        variance_agent = self.get_variance_for_given_thetas()
        Q, V, pi_s = self.solver.soft_valueIteration(
            self.env, dict(reward=reward_agent, variance=variance_agent)
        )
        _, mu, nu = self.solver.computeFeatureSVF_bellmann_averaged(self.env, pi_s)
        return (
            mu[: self.env.n_features_reward],
            nu[: self.env.n_features_reward, : self.env.n_features_reward],
            mu,
            nu,
        )

    def batch_MCE(self, verbose: bool = True):
        """
        implementation of Algorithm 1

        computes dual ascent of soft value iteration of gradient descent of the thetas until convergence

        Parameters
        ----------
        verbose : bool
            whether or not intermediate output should be displayed (by default true)
        """

        calc_theta_v = isinstance(self.solver, MDPSolver.MDPSolverVariance)

        theta_e_pos = np.zeros(self.env.n_features_reward)
        theta_e_neg = np.zeros(self.env.n_features_reward)
        self.theta_e = theta_e_pos - theta_e_neg

        if calc_theta_v:
            theta_v_pos = np.zeros(
                (self.env.n_features_reward, self.env.n_features_reward)
            )
            theta_v_neg = np.zeros(
                (self.env.n_features_reward, self.env.n_features_reward)
            )
            self.theta_v = theta_v_pos - theta_v_neg

        mu_reward_agent, mu_variance_agent, _, _ = self.get_mu_soft()

        if verbose:
            print("\n========== batch_MCE for " + self.agent_name + " =======")

        gradientconstant = 1
        t = 1
        while True:
            # set learning rate
            eta = gradientconstant / np.sqrt(t)

            if verbose:
                print("t=", t)
                print("...eta=", eta)
                print(
                    "...mu_reward_agent=",
                    mu_reward_agent,
                    " mu_demonstrator=",
                    self.mu_demonstrator[0],
                )
                print("...theta_e=", self.theta_e)
                print("...theta_v=", self.theta_v)

            # update lambda
            theta_e_pos_old = copy.deepcopy(theta_e_pos)
            theta_e_neg_old = copy.deepcopy(theta_e_neg)
            theta_e_old = copy.deepcopy(self.theta_e)

            if calc_theta_v:
                theta_v_pos_old = copy.deepcopy(theta_v_pos)
                theta_v_neg_old = copy.deepcopy(theta_v_neg)
                theta_v_old = copy.deepcopy(self.theta_v)

            theta_e_pos = theta_e_pos_old - eta * (
                mu_reward_agent - self.mu_demonstrator[0]
            )
            theta_e_neg = theta_e_neg_old - eta * (
                self.mu_demonstrator[0] - mu_reward_agent
            )

            if calc_theta_v:
                theta_v_pos = theta_v_pos_old - eta * (
                    mu_variance_agent - self.mu_demonstrator[1]
                )
                theta_v_neg = theta_v_neg_old - eta * (
                    self.mu_demonstrator[1] - mu_variance_agent
                )

            theta_e_pos = np.maximum(theta_e_pos, 0)
            theta_e_neg = np.maximum(theta_e_neg, 0)
            theta_e_pos = np.minimum(theta_e_pos, self.theta_upperBound)
            theta_e_neg = np.minimum(theta_e_neg, self.theta_upperBound)
            self.theta_e = theta_e_pos - theta_e_neg

            if calc_theta_v:
                theta_v_pos = np.maximum(theta_v_pos, 0)
                theta_v_neg = np.maximum(theta_v_neg, 0)
                theta_v_pos = np.minimum(theta_v_pos, self.theta_upperBound)
                theta_v_neg = np.minimum(theta_v_neg, self.theta_upperBound)
                self.theta_v = theta_v_pos - theta_v_neg

            # update state
            mu_reward_agent, mu_variance_agent, _, _ = self.get_mu_soft()

            diff_L2_norm_theta_e = np.linalg.norm(theta_e_old - self.theta_e)

            if calc_theta_v:
                diff_L2_norm_theta_v = np.linalg.norm(theta_v_old - self.theta_v)

            if verbose:
                print("...diff_L2_norm_theta_e=", diff_L2_norm_theta_e)
                if calc_theta_v:
                    print("...diff_L2_norm_theta_v=", diff_L2_norm_theta_v)

            # decide whether to continue
            if calc_theta_v:
                if (diff_L2_norm_theta_e < self.tol) and (
                    diff_L2_norm_theta_v < self.tol
                ):
                    if t >= self.miniter:
                        break
            elif diff_L2_norm_theta_e < self.tol:
                if t >= self.miniter:
                    break

            if t > self.maxiter:
                break

            t += 1
