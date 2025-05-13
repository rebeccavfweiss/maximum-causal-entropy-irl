import MDPSolver
import numpy as np
from environments.environment import Environment
from policy import TabularPolicy
from agents.agent import Agent
from time import time
import torch

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

        # TODO check which variables not used
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
        _, mu, nu = self.solver.compute_feature_SVF_bellmann_averaged(
            self.env, pi_agent
        )
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
        runtime = []

        initial_lr = 1.0
        min_lr = 0.01
        gamma = 0.99

        # Initialize PyTorch tensors for thetas
        theta_e = torch.zeros(self.env.n_features, requires_grad=True)
        optimizer_e = torch.optim.Adam([theta_e], lr=initial_lr)
        lr_lambda = lambda step: max(gamma ** np.log(step + 1), min_lr / initial_lr)
        scheduler_e = torch.optim.lr_scheduler.LambdaLR(
            optimizer_e, lr_lambda=lr_lambda
        )

        if calc_theta_v:
            theta_v = torch.zeros(
                (self.env.n_features, self.env.n_features), requires_grad=True
            )
            optimizer_v = torch.optim.Adam([theta_v], lr=initial_lr)
            lr_lambda = lambda step: max(gamma ** np.log(step + 1), min_lr / initial_lr)
            scheduler_v = torch.optim.lr_scheduler.LambdaLR(
                optimizer_v, lr_lambda=lr_lambda
            )

        mu_reward_agent, mu_variance_agent = self.get_mu_soft()

        if verbose:
            print("\n========== batch_MCE for " + self.agent_name + " =======")

        t = 1
        while True:
            start = time()

            # Get current theta values into the object (for get_mu_soft to use them)
            self.theta_e = theta_e.detach().numpy()
            if calc_theta_v:
                self.theta_v = theta_v.detach().numpy()

            # Recompute agent feature expectations
            mu_reward_agent, mu_variance_agent = self.get_mu_soft()

            # Compute gradient for reward part
            grad_e = torch.tensor(
                mu_reward_agent - self.mu_demonstrator[0], dtype=torch.float32
            )

            optimizer_e.zero_grad()
            theta_e.grad = grad_e
            optimizer_e.step()
            scheduler_e.step()

            # Clamp values (optional, depending on your upper bounds)
            with torch.no_grad():
                theta_e.clamp_(-self.theta_upperBound, self.theta_upperBound)

            if calc_theta_v:
                grad_v = torch.tensor(
                    mu_variance_agent - self.mu_demonstrator[1], dtype=torch.float32
                )
                optimizer_v.zero_grad()
                theta_v.grad = grad_v
                optimizer_v.step()
                scheduler_v.step()

                with torch.no_grad():
                    theta_v.clamp_(-self.theta_upperBound, self.theta_upperBound)

            end = time()
            runtime.append(end - start)

            # Convergence check
            theta_e_diff = torch.norm(theta_e.grad).item()
            if calc_theta_v:
                theta_v_diff = torch.norm(theta_v.grad).item()

            if verbose:
                print(
                    f"t={t}, lr={scheduler_e.get_last_lr()}, theta_e_diff={theta_e_diff}"
                )
                if calc_theta_v:
                    print(f"theta_v_diff={theta_v_diff}")

            if theta_e_diff < self.tol and (
                not calc_theta_v or theta_v_diff < 5 * self.tol
            ):
                if t >= self.miniter:
                    break

            if t > self.maxiter:
                break

            t += 1

        # Final assignment back to numpy
        self.theta_e = theta_e.detach().numpy()
        if calc_theta_v:
            self.theta_v = theta_v.detach().numpy()

        if verbose:
            print(f"Terminated in {t} iterations")

        return t, runtime
