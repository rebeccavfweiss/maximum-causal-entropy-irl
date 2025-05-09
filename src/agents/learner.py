import MDPSolver
import numpy as np
import copy
from environments.environment import Environment
from policy import TabularPolicy
from agents.agent import Agent
from time import time

_largenum = 1000000


class Learner(Agent):
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
        self.theta_e = np.zeros(self.env.n_features)
        self.theta_v = np.zeros((self.env.n_features, self.env.n_features))

        self.tol = config_agent["tol"]
        self.maxiter = config_agent["maxiter"]
        self.miniter = config_agent["miniter"]

        self.theta_upperBound = _largenum

        self.V = None
        self.policy = None
        self.reward = None
        self.solver = solver
        self.T = self.solver.T

        self.agent_name = agent_name

    def compute_and_draw(
        self, show: bool = False, store: bool = False, fignum: int = 0
    ) -> None:
        """
        computes soft_value iteration for given thetas and policy based on the result and draws policy

        Parameters
        ----------
        show : bool
            whether or not the plots should be shown
        store : bool
            whether or not the plots should be stored
        fignum : int
            identifier number for figure
        """
        pi_agent = self.compute_policy()
        self.policy = TabularPolicy(pi_agent)

        self.V = self.solver.compute_value_function_bellmann_averaged(
            self.env,
            self.policy.pi,
            dict(reward=self.env.reward),
        )  # compute the value function w.r.t to true reward parameters

        self.draw(show, store, fignum)

    def compute_policy(self):
        """
        Helper function to compute policy via SVI for the given reward parameters

        Returns
        -------
        pi_agent : ndarray
            tabular policy
        """
        self.reward = self.get_linear_reward()
        self.variance = self.get_variance()
        _, _, pi_agent = self.solver.soft_value_iteration(
            self.env, dict(reward=self.reward, variance=self.variance)
        )
        
        return pi_agent


    def get_linear_reward(self) -> np.ndarray:
        """
        computes the reward based on theta_e for every state

        Returns
        -------
        reward : numpy.ndarray
        """

        return self.env.get_reward_for_given_theta(self.theta_e)

    def get_variance(self) -> np.ndarray:
        """
        computes the variance term for every state needed for soft value iteration based on theta_v

        Returns
        -------
        variance : numpy.ndarray
        """

        return self.env.get_variance_for_given_theta(self.theta_v)

    def get_mu_soft(self) -> tuple[np.ndarray, np.ndarray]:
        """
        computes feature expectation and variance terms based on soft value iteration and computing the corresponding value function

        Returns
        -------
        feature expectation and variance, once restricted to the reward features, once the full arrays
        """
        pi_agent = self.compute_policy()
        _, mu, nu = self.solver.compute_feature_SVF_bellmann_averaged(self.env, pi_agent)
        return (
            mu,
            nu,
        )

    def batch_MCE(self, verbose: bool = True) -> tuple[int, list[float]]:
        """
        implementation of Algorithm 1

        computes dual ascent of soft value iteration of gradient descent of the thetas until convergence

        Parameters
        ----------
        verbose : bool
            whether or not intermediate output should be displayed (by default true)

        Returns
        -------
        int
            number of iterations used until convergence
        list[float]
            time used per iteration
        """

        calc_theta_v = isinstance(self.solver, MDPSolver.MDPSolverVariance)

        theta_e_pos = np.zeros(self.env.n_features)
        theta_e_neg = np.zeros(self.env.n_features)
        self.theta_e = np.zeros(self.env.n_features)

        runtime = []

        if calc_theta_v:
            theta_v_pos = np.zeros((self.env.n_features, self.env.n_features))
            theta_v_neg = np.zeros((self.env.n_features, self.env.n_features))
            self.theta_v = theta_v_pos - theta_v_neg

        mu_reward_agent, mu_variance_agent = self.get_mu_soft()

        if verbose:
            print("\n========== batch_MCE for " + self.agent_name + " =======")

        gradientconstant = 1
        t = 1
        while True:
            # set learning rate

            start = time()

            eta = gradientconstant / np.sqrt(t)

            if verbose:
                print("t=", t)
                print("...eta_e=", eta)
                print(
                    "...mu_reward_agent=",
                    mu_reward_agent,
                    " mu_variance_agent=",
                    mu_variance_agent,
                    " mu_demonstrator=",
                    self.mu_demonstrator[0],
                    " nu_demonstrator=",
                    self.mu_demonstrator[1],
                )
                print("...theta_e=", self.theta_e)
                print("...theta_v=", self.theta_v)

            theta_e_pos_old = copy.deepcopy(theta_e_pos)
            theta_e_neg_old = copy.deepcopy(theta_e_neg)
            theta_e_old = copy.deepcopy(self.theta_e)
            # update lambda

            theta_e_pos_old = copy.deepcopy(theta_e_pos)
            theta_e_neg_old = copy.deepcopy(theta_e_neg)
            theta_e_old = copy.deepcopy(self.theta_e)

            theta_e_pos = theta_e_pos_old - eta * (
                mu_reward_agent - self.mu_demonstrator[0]
            )
            theta_e_neg = theta_e_neg_old - eta * (
                self.mu_demonstrator[0] - mu_reward_agent
            )

            theta_e_pos = np.maximum(theta_e_pos, 0)
            theta_e_neg = np.maximum(theta_e_neg, 0)
            theta_e_pos = np.minimum(theta_e_pos, self.theta_upperBound)
            theta_e_neg = np.minimum(theta_e_neg, self.theta_upperBound)
            self.theta_e = theta_e_pos - theta_e_neg

            if calc_theta_v:
                theta_v_pos_old = copy.deepcopy(theta_v_pos)
                theta_v_neg_old = copy.deepcopy(theta_v_neg)
                theta_v_old = copy.deepcopy(self.theta_v)

                theta_v_pos = theta_v_pos_old - eta * (
                    mu_variance_agent - self.mu_demonstrator[1]
                )
                theta_v_neg = theta_v_neg_old - eta * (
                    self.mu_demonstrator[1] - mu_variance_agent
                )

                theta_v_pos = np.maximum(theta_v_pos, 0)
                theta_v_neg = np.maximum(theta_v_neg, 0)
                theta_v_pos = np.minimum(theta_v_pos, self.theta_upperBound)
                theta_v_neg = np.minimum(theta_v_neg, self.theta_upperBound)
                self.theta_v = theta_v_pos - theta_v_neg

            # update state
            mu_reward_agent, mu_variance_agent = self.get_mu_soft()

            diff_L2_norm_theta_e = np.linalg.norm(theta_e_old - self.theta_e)

            if calc_theta_v:
                diff_L2_norm_theta_v = np.linalg.norm(theta_v_old - self.theta_v)

            end = time()

            runtime.append(end - start)

            if verbose:
                print("...diff_L2_norm_theta_e=", diff_L2_norm_theta_e)
                if calc_theta_v:
                    print("...diff_L2_norm_theta_v=", diff_L2_norm_theta_v)

            # decide whether to continue
            if calc_theta_v:
                if (diff_L2_norm_theta_e < self.tol) and (
                    diff_L2_norm_theta_v < 5 * self.tol
                ):
                    if t >= self.miniter:
                        break
            elif diff_L2_norm_theta_e < self.tol:
                if t >= self.miniter:
                    break

            if t > self.maxiter:
                break

            t += 1

        if verbose:
            print(f"Terminated in {t} iterations")

        return t, runtime
