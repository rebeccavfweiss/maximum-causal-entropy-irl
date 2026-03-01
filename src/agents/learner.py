import solvers.MDP_solver as MDP_solver
from solvers.MDP_solver_exact import MDPSolverExact, MDPSolverExactVariance
from solvers.MDP_solver_approximation import (
    MDPSolverApproximation,
    MDPSolverApproximationVariance,
)
import numpy as np
import wandb
from environments.environment import Environment, GridEnvironment
from environments.car_racing_environment import CarRacingEnvironment
from environments.object_world_environment import ObjectWorldEnvironment
from policy import Policy
from agents.agent import Agent
from time import time
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from abc import abstractmethod
from pathlib import Path

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
        tol_exp : convergence tolerance for batch_MCE
        tol_var : convergence tolerance for batch_MCE
        maxiter : maximal number of iterations for batch_MCE
        miniter : minimal number of iterations for batch_MCE
    agent_name : str
        name of the agent
    solver : MDPSolver.MDPSolver
        solver to use (either only expectation matching or also variance matching)
    learning_rate_e
        custom learning rate function for MCE IRL for theta_e
    learning_rate_v
        custom learning rate function for MCE IRL for theta_v
    optimizer_e:
        optimizer to use during the dual ascent for theta_e
    optimizer_v:
        optimizer to use during the dual acent for theta_v
    optimizer_e_kwargs:
        keyed arguments to use for the optimizer of theta_e
    optimizer_v_kwargs:
        keyed arguments to use for the optimizer of theta_v
    """

    def __init__(
        self,
        env: Environment,
        mu_demonstrator: tuple[float, float],
        config_agent: dict[str:any],
        agent_name: str,
        solver: MDP_solver.MDPSolver,
        learning_rate_e=None,
        learning_rate_v=None,
        optimizer_e=None,
        optimizer_v=None,
        optimizer_e_kwargs=None,
        optimizer_v_kwargs=None,
    ):

        super().__init__(env, agent_name)
        self.mu_demonstrator = mu_demonstrator
        self.theta_e = np.zeros(self.env.n_features)
        self.theta_v = np.zeros((self.env.n_features, self.env.n_features))

        self.tol_exp = config_agent["tol_exp"]
        self.tol_var = config_agent.get("tol_var", None)
        self.maxiter = config_agent["maxiter"]
        self.miniter = config_agent["miniter"]
        self.n_trajectories = config_agent.get("n_trajectories", None)

        self.learning_rate_e = learning_rate_e
        self.learning_rate_v = learning_rate_v

        self.optimizer_e = optimizer_e
        self.optimizer_v = optimizer_v

        self.optimizer_e_kwargs = optimizer_e_kwargs
        self.optimizer_v_kwargs = optimizer_v_kwargs

        self.theta_upperBound = _largenum

        self.V = None
        self.policy = None
        self.reward = None
        self.solver = solver
        self.T = self.solver.T

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

        self.policy = self.compute_policy()

        self.V = self.solver.compute_value_function_bellmann_averaged(
            self.env,
            self.policy,
            dict(reward=self.env.reward),
        )  # compute the value function w.r.t to true reward parameters

        path_to_file = self.render(show, store, fignum)
        if path_to_file is not None:
            # log a video to see how the current policy is doing
            wandb.log(
                {
                    f"final/video_{self.agent_name}": wandb.Video(
                        path_to_file, fps=4, format="mp4"
                    )
                }
            )

    def compute_policy(self) -> Policy:
        """
        Helper function to compute policy via SVI for the given reward parameters

        Returns
        -------
        pi_agent : ndarray
            tabular policy
        """
        self.reward = self.get_linear_reward()
        self.variance = self.get_variance()
        return self.solver.soft_value_iteration(
            self.env, dict(reward=self.reward, variance=self.variance)
        )

    @abstractmethod
    def get_linear_reward(self) -> any:
        pass

    @abstractmethod
    def get_variance(self) -> any:
        pass

    def get_mu_soft(self) -> tuple[np.ndarray, np.ndarray]:
        """
        computes feature expectation and variance terms based on soft value iteration and computing the corresponding value function

        Returns
        -------
        feature expectation and variance, trained policy
        """
        self.policy = self.compute_policy()

        return self.solver.compute_feature_SVF_bellmann_averaged(
            self.env, self.policy, self.n_trajectories
        )

    def batch_MCE(
        self, alternate_every: int = None, var_factor: int = 2
    ) -> tuple[int, list[float]]:
        """
        implementation of Algorithm 1

        computes dual ascent of soft value iteration of gradient descent of the thetas until convergence

        Returns
        -------
        int
            number of iterations used until convergence
        list[float]
            time used per iteration
        """

        calc_theta_v = isinstance(self.solver, MDPSolverExactVariance) or isinstance(
            self.solver, MDPSolverApproximationVariance
        )
        runtime = []

        min_lr = 0.001
        gamma = 0.99
        theta_e_diff = np.inf
        theta_v_diff = np.inf

        # Initialize PyTorch tensors for thetas
        theta_e = torch.zeros(self.env.n_features, requires_grad=True)
        if self.optimizer_e is None:
            optimizer_e = torch.optim.Adam([theta_e], lr=1.0)
        else:
            optimizer_e = self.optimizer_e([theta_e], **self.optimizer_e_kwargs)
        if self.learning_rate_e is None:
            lr_lambda_e = lambda step: max(gamma ** np.log(step + 1), min_lr / 1.0)

            scheduler_e = torch.optim.lr_scheduler.LambdaLR(
                optimizer_e, lr_lambda=lr_lambda_e
            )
        else:

            scheduler_e = self.learning_rate_e["scheduler"](
                optimizer_e, **self.learning_rate_e["scheduler_kwargs"]
            )

        if calc_theta_v:
            theta_v = torch.zeros(
                (self.env.n_features, self.env.n_features), requires_grad=True
            )
            if self.optimizer_v is None:
                optimizer_v = torch.optim.Adam([theta_v], lr=1.0, eps=1e-7)
            else:
                optimizer_v = self.optimizer_v([theta_v], **self.optimizer_v_kwargs)
            if self.learning_rate_v is None:
                lr_lambda_v = lambda step: max(gamma ** np.log(step + 1), min_lr / 1.0)

                scheduler_v = torch.optim.lr_scheduler.LambdaLR(
                    optimizer_v, lr_lambda=lr_lambda_v
                )
            else:

                scheduler_v = self.learning_rate_v["scheduler"](
                    optimizer_v, **self.learning_rate_v["scheduler_kwargs"]
                )

        if isinstance(self.env, ObjectWorldEnvironment):
            # for object world track reward evolution
            self.rewards = []
            current_rewards = self.env.get_reward_for_given_theta(self.theta_e)
            if calc_theta_v:
                current_rewards += self.env.get_variance_for_given_theta(self.theta_v)
            self.rewards.append(current_rewards)

        t = 1
        while True:
            start = time()

            # Get current theta values into the object (for get_mu_soft to use them)
            self.theta_e = theta_e.detach().numpy()
            if calc_theta_v:
                self.theta_v = theta_v.detach().numpy()

            # Recompute agent feature expectations
            mu_reward_agent, mu_variance_agent = self.get_mu_soft()

            if (
                (t % 10 - 1) == 0
                and isinstance(self.solver, MDPSolverApproximation)
                # and isinstance(self.env, CarRacingEnvironment)
            ):
                # evaluate agent with a recorded episode
                path_to_file = self.render(False, True)
                if path_to_file is not None:
                    # log a video to see how the current policy is doing
                    wandb.log(
                        {
                            f"eval/video_{self.agent_name}": wandb.Video(
                                path_to_file, fps=4, format="mp4"
                            )
                        }
                    )
            if (alternate_every is None) or (
                int(t / alternate_every) % var_factor == 0
            ):
                # Compute gradient for reward part
                grad_e = torch.tensor(
                    mu_reward_agent - self.mu_demonstrator[0], dtype=torch.float32
                )

                optimizer_e.zero_grad()
                theta_e.grad = grad_e
                optimizer_e.step()
                if isinstance(scheduler_e, ReduceLROnPlateau):
                    scheduler_e.step(theta_e_diff)
                else:
                    scheduler_e.step()

                # Clamp values (optional, depending on your upper bounds)
                with torch.no_grad():
                    theta_e.clamp_(-self.theta_upperBound, self.theta_upperBound)

            if calc_theta_v and (
                (alternate_every is None)
                or (int(t / alternate_every) % var_factor != 0)
            ):
                grad_v = torch.tensor(
                    mu_variance_agent - self.mu_demonstrator[1], dtype=torch.float32
                )
                optimizer_v.zero_grad()
                theta_v.grad = grad_v
                optimizer_v.step()
                if isinstance(scheduler_v, ReduceLROnPlateau):
                    scheduler_v.step(theta_v_diff)
                else:
                    scheduler_v.step()

                with torch.no_grad():
                    theta_v.clamp_(-self.theta_upperBound, self.theta_upperBound)

            end = time()
            runtime.append(end - start)

            # Convergence check

            if (alternate_every is None) or (
                int(t / alternate_every) % var_factor == 0
            ):
                theta_e_diff = torch.norm(theta_e.grad).item()
            if calc_theta_v and (
                (alternate_every is None)
                or (int(t / alternate_every) % var_factor != 0)
            ):
                theta_v_diff = torch.norm(theta_v.grad).item()

            wandb.log(
                {
                    f"step_{self.agent_name}": t,
                    f"theta_e_diff_{self.agent_name}": theta_e_diff,
                    f"lr_e_{self.agent_name}": scheduler_e.get_last_lr()[0],
                    **(
                        {
                            f"theta_v_diff_{self.agent_name}": theta_v_diff,
                            f"lr_v_{self.agent_name}": scheduler_v.get_last_lr()[0],
                        }
                        if calc_theta_v
                        else {}
                    ),
                }
            )

            if isinstance(self.env, ObjectWorldEnvironment):
                current_rewards = self.env.get_reward_for_given_theta(self.theta_e)
                if calc_theta_v:
                    current_rewards += self.env.get_variance_for_given_theta(
                        self.theta_v
                    )
                self.rewards.append(current_rewards)

            if theta_e_diff < self.tol_exp and (
                not calc_theta_v or theta_v_diff < self.tol_var
            ):
                if t >= self.miniter:
                    break

            if t >= self.maxiter:
                break

            t += 1

        # Final assignment back to numpy
        self.theta_e = theta_e.detach().numpy()
        if calc_theta_v:
            self.theta_v = theta_v.detach().numpy()

        return t, runtime


class TabularLearner(Learner):
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
        tol_exp : convergence tolerance for batch_MCE
        tol_var : convergence tolerance for batch_MCE
        maxiter : maximal number of iterations for batch_MCE
        miniter : minimal number of iterations for batch_MCE
    agent_name : str
        name of the agent
    solver : MDPSolverExact
        solver to use (either only expectation matching or also variance matching) (must be a tabular/exact solver)
    learning_rate_e
        custom learning rate (decreasing) function for theta_e
    learning_rate_v
        custom learning rate (decreasing) function for theta_v
    optimizer_e:
        optimizer to use during the dual ascent for theta_e
    optimizer_v:
        optimizer to use during the dual acent for theta_v
    optimizer_e_kwargs:
        keyed arguments to use for the optimizer of theta_e
    optimizer_v_kwargs:
        keyed arguments to use for the optimizer of theta_v
    """

    def __init__(
        self,
        env: GridEnvironment,
        mu_demonstrator: tuple[float, float],
        config_agent: dict[str:any],
        agent_name: str,
        solver: MDPSolverExact,
        learning_rate_e=None,
        learning_rate_v=None,
        optimizer_e=None,
        optimizer_v=None,
        optimizer_e_kwargs=None,
        optimizer_v_kwargs=None,
    ):

        super().__init__(
            env,
            mu_demonstrator,
            config_agent,
            agent_name,
            solver,
            learning_rate_e,
            learning_rate_v,
            optimizer_e,
            optimizer_v,
            optimizer_e_kwargs,
            optimizer_v_kwargs,
        )

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

    def render(self, show: bool = False, store: bool = False, fignum: int = 0) -> Path:
        """
        Overwrite base rendering function in case we have an object world environment to include the reward evolution

        Parameters
        ----------
        show : bool
            whether or not the plot should be shown
        store : bool
            whether or not the plot should be stored
        fignum : int
            identifier number for the figure

        Returns
        -------
        path : Path
            path to the stored video (for the car racing environment) and None else
        """
        if isinstance(self.env, ObjectWorldEnvironment):
            return self.env.render(
                self.rewards, self.V, self.policy, self.agent_name, store, show
            )

        else:
            return super().render(show, store, fignum)


class ApproximateLearner(Learner):
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
        tol_exp : convergence tolerance for batch_MCE
        tol_var : convergence tolerance for batch_MCE
        maxiter : maximal number of iterations for batch_MCE
        miniter : minimal number of iterations for batch_MCE
    agent_name : str
        name of the agent
    solver : MDPSolverExact
        solver to use (either only expectation matching or also variance matching) (must be an approximation solver)
    learning_rate_e
        custom learning rate (decreasing) function for theta_e
    learning_rate_v
        custom learning rate (decreasing) function for theta_v
    optimizer_e:
        optimizer to use during the dual ascent for theta_e
    optimizer_v:
        optimizer to use during the dual acent for theta_v
    optimizer_e_kwargs:
        keyed arguments to use for the optimizer of theta_e
    optimizer_v_kwargs:
        keyed arguments to use for the optimizer of theta_v
    heuristic_theta_e : ndarray
        heuristic that should be used for theta_e to simplify training
    heuristic_theta_v : ndarray
        heuristic that should be used for theta_v to simplify training
    """

    def __init__(
        self,
        env: Environment,
        mu_demonstrator: tuple[float, float],
        config_agent: dict[str:any],
        agent_name: str,
        solver: MDPSolverApproximation,
        learning_rate_e=None,
        learning_rate_v=None,
        optimizer_e=None,
        optimizer_v=None,
        optimizer_e_kwargs=None,
        optimizer_v_kwargs=None,
        heuristic_theta_e: np.ndarray = None,
        heuristic_theta_v: np.ndarray = None,
    ):

        super().__init__(
            env,
            mu_demonstrator,
            config_agent,
            agent_name,
            solver,
            learning_rate_e,
            learning_rate_v,
            optimizer_e,
            optimizer_v,
            optimizer_e_kwargs,
            optimizer_v_kwargs,
        )

        if heuristic_theta_e is not None:
            assert (
                heuristic_theta_e.shape[0] == self.env.n_features
            ), f"heuristic for theta_e has wrong dimension(s), expected {self.env.n_features} got {heuristic_theta_e.shape}"

            self.theta_e = heuristic_theta_e

        if heuristic_theta_v is not None:
            assert heuristic_theta_v.shape == (
                self.env.n_features,
                self.env.n_features,
            ), f"heuristic for theta_v has wrong dimension(s), expexted {(self.env.n_features, self.env.n_features)} got {heuristic_theta_v.shape}"

            self.theta_v = heuristic_theta_v

    def get_linear_reward(self) -> any:
        """
        creates a linear reward function w.r.t state observations based on theta_e

        Returns
        -------
        reward : function
        """

        return lambda state: self.theta_e.dot(state)

    def get_variance(self) -> any:
        """
        creates a quadratic function w.r.t state observations based on theta_v

        Returns
        -------
        variance : function
        """

        return lambda state: (self.theta_v.dot(state)).dot(state)
