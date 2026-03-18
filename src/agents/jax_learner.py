"""
JAX-based approximate learner with warm-starting support.

Overrides batch_MCE to use optax for theta optimization and
pass/receive network parameters between IRL iterations for
warm-starting the inner RL loop.
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
import wandb
from time import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

from agents.learner import ApproximateLearner
from solvers.MDP_solver_jax import JaxSolverVariance
from solvers.MDP_solver_approximation import MDPSolverApproximation
from environments.environment import Environment
from policy import Policy
import solvers.MDP_solver as MDP_solver


class JaxApproximateLearner(ApproximateLearner):
    """
    ApproximateLearner with JAX-based inner loop and warm-starting.

    Key differences from ApproximateLearner:
    1. Inner RL loop uses JAX DQN/SAC (via JaxSolver) instead of SB3
    2. Warm-starting: reuses previous iteration's network weights
    3. Theta optimization uses optax (JAX) instead of torch.optim
    4. First iteration trains fully, subsequent iterations fine-tune

    Parameters
    ----------
    env : Environment
        The environment
    mu_demonstrator : tuple[float, float]
        Feature expectation and variance terms of the demonstrator
    config_agent : dict
        Configuration parameters (tol_exp, tol_var, maxiter, miniter, etc.)
    agent_name : str
        Name of the agent
    solver : JaxSolver
        JAX-based solver (JaxSolverExpectation or JaxSolverVariance)
    optax_optimizer_e : optax optimizer
        Optax optimizer for theta_e (e.g., optax.adam(lr))
    optax_optimizer_v : optax optimizer or None
        Optax optimizer for theta_v
    heuristic_theta_e : ndarray or None
        Initial theta_e heuristic
    heuristic_theta_v : ndarray or None
        Initial theta_v heuristic
    lr_decay_rate_e : float
        Learning rate decay rate for theta_e
    lr_decay_rate_v : float
        Learning rate decay rate for theta_v
    """

    def __init__(
        self,
        env: Environment,
        mu_demonstrator: tuple[float, float],
        config_agent: dict,
        agent_name: str,
        solver: MDP_solver.MDPSolver,
        optax_optimizer_e=None,
        optax_optimizer_v=None,
        heuristic_theta_e: np.ndarray = None,
        heuristic_theta_v: np.ndarray = None,
        lr_e: float = 0.1,
        lr_v: float = 0.05,
        lr_decay_rate_e: float = 0.95,
        lr_decay_rate_v: float = 0.9,
    ):
        # Initialize base Learner via ApproximateLearner, but we won't use
        # its torch-based optimizer infrastructure
        super().__init__(
            env=env,
            mu_demonstrator=mu_demonstrator,
            config_agent=config_agent,
            agent_name=agent_name,
            solver=solver,
            learning_rate_e=None,
            learning_rate_v=None,
            optimizer_e=None,
            optimizer_v=None,
            optimizer_e_kwargs=None,
            optimizer_v_kwargs=None,
            heuristic_theta_e=heuristic_theta_e,
            heuristic_theta_v=heuristic_theta_v,
        )

        # Override with optax optimizers
        self._optax_optimizer_e = optax_optimizer_e or optax.adam(lr_e)
        self._optax_optimizer_v = optax_optimizer_v or optax.adam(lr_v)
        self._lr_decay_rate_e = lr_decay_rate_e
        self._lr_decay_rate_v = lr_decay_rate_v

    def batch_MCE(
        self,
        alternate_every: int = None,
        var_factor: int = 2,
        early_stop_window: int = 200,
    ) -> tuple[int, list[float]]:
        """
        Modified batch MCE with warm-starting, optax optimization,
        and early stopping.

        Key changes from parent:
        - solver.soft_value_iteration returns (policy, train_state)
        - train_state is passed back as prev_params for warm-starting
        - Theta optimization uses optax instead of torch
        - Early stopping: if loss trend is non-decreasing over the last
          `early_stop_window` iterations, training stops

        Returns
        -------
        int
            Number of iterations used until convergence
        list[float]
            Time used per iteration
        """
        calc_theta_v = isinstance(self.solver, JaxSolverVariance)
        runtime = []

        theta_e_diff = np.inf
        theta_v_diff = np.inf

        # History for early stopping
        loss_history_e = []
        loss_history_v = []

        # Initialize theta_e as JAX array
        if self.theta_e is not None and np.any(self.theta_e != 0):
            theta_e = jnp.array(self.theta_e, dtype=jnp.float32)
        else:
            theta_e = jnp.zeros(self.env.n_features, dtype=jnp.float32)

        optimizer_e = self._optax_optimizer_e
        opt_state_e = optimizer_e.init(theta_e)

        if calc_theta_v:
            if self.theta_v is not None and np.any(self.theta_v != 0):
                theta_v = jnp.array(self.theta_v, dtype=jnp.float32)
            else:
                theta_v = jnp.zeros(
                    (self.env.n_features, self.env.n_features), dtype=jnp.float32
                )
            optimizer_v = self._optax_optimizer_v
            opt_state_v = optimizer_v.init(theta_v)

        prev_train_state = None  # No warm-start on first iteration
        t = 1

        while True:
            start = time()

            # Sync theta values to numpy for the solver's reward functions
            self.theta_e = np.array(theta_e)
            if calc_theta_v:
                self.theta_v = np.array(theta_v)

            # Compute reward and variance functions
            self.reward = self.get_linear_reward()
            self.variance = self.get_variance()

            # KEY CHANGE: solver returns (policy, train_state) for warm-starting
            self.policy, prev_train_state = self.solver.soft_value_iteration(
                self.env,
                dict(reward=self.reward, variance=self.variance),
                prev_params=prev_train_state,
            )

            # Compute feature expectations using trajectories
            mu_reward_agent, mu_variance_agent = (
                self.solver.compute_feature_SVF_bellmann_averaged(
                    self.env, self.policy, self.n_trajectories
                )
            )

            # Update theta_e
            if (alternate_every is None) or (
                int(t / alternate_every) % var_factor == 0
            ):
                grad_e = jnp.array(
                    mu_reward_agent - self.mu_demonstrator[0], dtype=jnp.float32
                )
                updates_e, opt_state_e = optimizer_e.update(
                    grad_e, opt_state_e, theta_e
                )
                theta_e = optax.apply_updates(theta_e, updates_e)
                theta_e = jnp.clip(
                    theta_e, -self.theta_upperBound, self.theta_upperBound
                )
                theta_e_diff = float(jnp.linalg.norm(grad_e))

            # Update theta_v
            if calc_theta_v and (
                (alternate_every is None)
                or (int(t / alternate_every) % var_factor != 0)
            ):
                grad_v = jnp.array(
                    mu_variance_agent - self.mu_demonstrator[1], dtype=jnp.float32
                )
                updates_v, opt_state_v = optimizer_v.update(
                    grad_v, opt_state_v, theta_v
                )
                theta_v = optax.apply_updates(theta_v, updates_v)
                theta_v = jnp.clip(
                    theta_v, -self.theta_upperBound, self.theta_upperBound
                )
                theta_v_diff = float(jnp.linalg.norm(grad_v))

            end = time()
            runtime.append(end - start)

            # Track loss history for early stopping
            loss_history_e.append(theta_e_diff)
            if calc_theta_v:
                loss_history_v.append(theta_v_diff)

            # Logging
            log_data = {
                f"step_{self.agent_name}": t,
                f"theta_e_diff_{self.agent_name}": theta_e_diff,
            }
            if calc_theta_v:
                log_data[f"theta_v_diff_{self.agent_name}"] = theta_v_diff

            # Log warm-start info
            if prev_train_state is not None:
                log_data[f"warm_started_{self.agent_name}"] = t > 1

            wandb.log(log_data)

            # Convergence check
            if theta_e_diff < self.tol_exp and (
                not calc_theta_v or theta_v_diff < self.tol_var
            ):
                if t >= self.miniter:
                    break

            if t >= self.maxiter:
                break

            # Early stopping: check if loss is stagnating
            if t >= self.miniter and t >= early_stop_window:
                e_stagnated = self._check_stagnation(loss_history_e, early_stop_window)
                v_stagnated = not calc_theta_v or self._check_stagnation(
                    loss_history_v, early_stop_window
                )
                if e_stagnated and v_stagnated:
                    wandb.log({f"early_stopped_{self.agent_name}": True})
                    break

            t += 1

        # Final assignment
        self.theta_e = np.array(theta_e)
        if calc_theta_v:
            self.theta_v = np.array(theta_v)

        return t, runtime

    def compute_policy(self) -> Policy:
        """
        Override to handle the (policy, train_state) return from JaxSolver.
        """
        self.reward = self.get_linear_reward()
        self.variance = self.get_variance()
        policy, _ = self.solver.soft_value_iteration(
            self.env, dict(reward=self.reward, variance=self.variance)
        )
        return policy
