"""
MMD-based MCE IRL learner.

Replaces explicit moment matching (E[phi], Var[phi]) with Maximum Mean
Discrepancy, implicitly matching ALL moments via a Gaussian RBF kernel.
The MMD witness function serves as the reward signal.
"""

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from time import time
from functools import partial

from agents.jax_learner import JaxApproximateLearner
from environments.environment import Environment
from policy import Policy
import solvers.MDP_solver as MDP_solver


class MMDLearner(JaxApproximateLearner):
    """
    MCE IRL learner that uses MMD for implicit all-moment matching.

    Instead of explicit theta_e/theta_v with moment-matching gradients,
    uses the MMD witness function as the reward signal. A Gaussian RBF
    kernel implicitly matches all moments of the feature distribution.

    Parameters
    ----------
    env : Environment
        The environment
    mu_demonstrator : tuple[float, float]
        Feature expectation and variance (used for parent init, not for MMD)
    config_agent : dict
        Configuration parameters
    agent_name : str
        Name of the agent
    solver : JaxSolver
        JAX-based solver
    expert_features : ndarray, shape (N, d)
        Per-trajectory feature sums from the expert demonstrator
    kernel_bandwidth : float or None
        RBF kernel bandwidth. None = use median heuristic
    tol_mmd : float
        Convergence threshold for MMD^2
    """

    def __init__(
        self,
        env: Environment,
        mu_demonstrator: tuple[float, float],
        config_agent: dict,
        agent_name: str,
        solver: MDP_solver.MDPSolver,
        expert_features: np.ndarray,
        kernel_bandwidth: float = None,
        tol_mmd: float = 0.01,
        **kwargs,
    ):
        super().__init__(
            env=env,
            mu_demonstrator=mu_demonstrator,
            config_agent=config_agent,
            agent_name=agent_name,
            solver=solver,
            **kwargs,
        )
        self.expert_features = jnp.array(expert_features, dtype=jnp.float32)
        self.tol_mmd = tol_mmd
        self.kernel_bandwidth = kernel_bandwidth

    @staticmethod
    def rbf_kernel(X, Y, bandwidth):
        """
        Compute RBF kernel matrix K[i,j] = exp(-||x_i - y_j||^2 / (2*sigma^2)).

        Parameters
        ----------
        X : jnp.array, shape (n, d)
        Y : jnp.array, shape (m, d)
        bandwidth : float

        Returns
        -------
        K : jnp.array, shape (n, m)
        """
        dists = jnp.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
        return jnp.exp(-dists / (2 * bandwidth**2))

    @staticmethod
    def compute_mmd_squared(X, Y, bandwidth):
        """
        Compute MMD^2 between two sets of feature vectors.

        Parameters
        ----------
        X : jnp.array, shape (n, d) - expert features
        Y : jnp.array, shape (m, d) - learner features
        bandwidth : float

        Returns
        -------
        mmd_sq : float
        """
        K_xx = MMDLearner.rbf_kernel(X, X, bandwidth)
        K_yy = MMDLearner.rbf_kernel(Y, Y, bandwidth)
        K_xy = MMDLearner.rbf_kernel(X, Y, bandwidth)
        return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()

    @staticmethod
    def median_heuristic(X, Y):
        """
        Compute kernel bandwidth via median heuristic.

        Parameters
        ----------
        X : jnp.array, shape (n, d)
        Y : jnp.array, shape (m, d)

        Returns
        -------
        bandwidth : float
        """
        combined = jnp.concatenate([X, Y], axis=0)
        dists = jnp.sum((combined[:, None, :] - combined[None, :, :]) ** 2, axis=-1)
        return jnp.sqrt(jnp.median(dists[dists > 0]) + 1e-8)

    def compute_witness_reward(self, learner_features):
        """
        Create reward function from MMD witness function.

        witness(x) = mean_i[k(x, expert_i)] - mean_j[k(x, learner_j)]

        This function is positive where the learner under-visits relative
        to the expert and negative where it over-visits.

        Parameters
        ----------
        learner_features : jnp.array, shape (m, d)

        Returns
        -------
        reward_fn : callable
            Function mapping state -> float reward
        bandwidth : float
            Kernel bandwidth used
        """
        bandwidth = self.kernel_bandwidth
        if bandwidth is None:
            bandwidth = float(self.median_heuristic(self.expert_features, learner_features))

        expert_feat = np.array(self.expert_features)
        learner_feat = np.array(learner_features)
        bw = float(bandwidth)

        def reward_fn(state):
            phi = np.array(state, dtype=np.float32).flatten()
            # k(phi, expert_i) for all i
            dists_expert = np.sum((phi - expert_feat) ** 2, axis=-1)
            k_expert = np.mean(np.exp(-dists_expert / (2 * bw**2)))
            # k(phi, learner_j) for all j
            dists_learner = np.sum((phi - learner_feat) ** 2, axis=-1)
            k_learner = np.mean(np.exp(-dists_learner / (2 * bw**2)))
            return float(k_expert - k_learner)

        return reward_fn, bandwidth

    def _collect_trajectory_features(self, n_trajectories: int) -> jnp.ndarray:
        """
        Collect per-trajectory discounted feature sums.

        Parameters
        ----------
        n_trajectories : int

        Returns
        -------
        features : jnp.array, shape (n_trajectories, n_features)
        """
        feature_list = []
        for _ in range(n_trajectories):
            trajectory = self.solver.generate_episode(
                self.env, self.policy, self.T
            )
            if len(trajectory) == 0:
                feature_list.append(np.zeros(self.env.n_features, dtype=np.float32))
                continue

            feat_sum = trajectory[0][0].flatten().astype(np.float32)
            for i in range(len(trajectory)):
                feat_sum += self.env.gamma ** (i + 1) * trajectory[i][2].flatten().astype(np.float32)
            feature_list.append(feat_sum)

        return jnp.array(feature_list)

    def batch_MCE(self, **kwargs) -> tuple[int, list[float]]:
        """
        Modified batch MCE using MMD witness function as reward.

        Key differences from standard batch_MCE:
        - No theta_e or theta_v parameters to optimize
        - Reward = witness function (non-parametric)
        - Convergence = MMD^2 < tol_mmd
        - First iteration uses zero reward (random exploration)

        Returns
        -------
        int
            Number of iterations
        list[float]
            Time per iteration
        """
        runtime = []
        prev_train_state = None
        learner_features = None
        t = 1

        n_traj = self.n_trajectories or 100

        while True:
            start = time()

            # Compute reward from MMD witness function
            if learner_features is None:
                # First iteration: zero reward (exploration)
                reward_fn = lambda s: 0.0
            else:
                reward_fn, bandwidth = self.compute_witness_reward(learner_features)

            self.reward = reward_fn
            self.variance = lambda s: 0.0

            # Train policy with warm-starting
            self.policy, prev_train_state = self.solver.soft_value_iteration(
                self.env,
                dict(reward=self.reward, variance=self.variance),
                prev_params=prev_train_state,
            )

            # Collect per-trajectory feature sums
            learner_features = self._collect_trajectory_features(n_traj)

            # Compute MMD^2
            bw = self.kernel_bandwidth
            if bw is None:
                bw = float(self.median_heuristic(self.expert_features, learner_features))

            mmd_sq = float(self.compute_mmd_squared(
                self.expert_features, learner_features, bw
            ))

            end = time()
            runtime.append(end - start)

            wandb.log({
                f"mmd_squared_{self.agent_name}": mmd_sq,
                f"step_{self.agent_name}": t,
                f"kernel_bandwidth_{self.agent_name}": float(bw),
                f"warm_started_{self.agent_name}": t > 1,
            })

            # Convergence check
            if mmd_sq < self.tol_mmd and t >= self.miniter:
                break
            if t >= self.maxiter:
                break

            t += 1

        # Fit linear reward for interpretability/comparison
        self.theta_e = self._fit_linear_reward(learner_features, bw)

        return t, runtime

    def _fit_linear_reward(self, learner_features, bandwidth):
        """
        Fit theta such that r(s) ~ theta * phi(s) approximates the witness function.

        Parameters
        ----------
        learner_features : jnp.array
        bandwidth : float

        Returns
        -------
        theta : ndarray
        """
        all_features = np.concatenate([
            np.array(self.expert_features),
            np.array(learner_features),
        ], axis=0)

        reward_fn, _ = self.compute_witness_reward(learner_features)
        witness_values = np.array([reward_fn(f) for f in all_features])

        theta, _, _, _ = np.linalg.lstsq(all_features, witness_values, rcond=None)
        return theta
