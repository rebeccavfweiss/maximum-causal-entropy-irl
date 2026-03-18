"""
Tabular MMD-based MCE IRL learner.

Uses Maximum Mean Discrepancy with tabular (exact) MDP solvers.
The MMD witness function serves as the reward signal, producing
per-state reward arrays for the exact solver.

Reuses kernel computations from MMDLearner (JAX-based) via static methods.
"""

import numpy as np
import jax.numpy as jnp
import wandb
from time import time

from agents.learner import TabularLearner
from agents.mmd_learner import MMDLearner
from environments.environment import GridEnvironment
from environments.object_world_environment import ObjectWorldEnvironment
from solvers.MDP_solver_exact import MDPSolverExactExpectation
import solvers.MDP_solver as MDP_solver


class TabularMMDLearner(TabularLearner):
    """
    MCE IRL learner using MMD with tabular (exact) MDP solvers.

    Instead of parametric theta_e/theta_v with moment-matching gradients,
    uses the MMD witness function evaluated at each state's feature vector
    to produce a per-state reward array for the exact solver.

    Parameters
    ----------
    env : GridEnvironment
        The environment (must have feature_matrix)
    mu_demonstrator : tuple[float, float]
        Feature expectation and variance (used for parent init, not for MMD)
    config_agent : dict
        Configuration parameters (maxiter, miniter, n_trajectories, etc.)
    agent_name : str
        Name of the agent
    solver : MDPSolverExactExpectation
        Exact MDP solver
    expert_features : ndarray, shape (N, d)
        Per-trajectory feature sums from the expert demonstrator
    kernel_bandwidth : float or None
        RBF kernel bandwidth. None = use median heuristic
    tol_mmd : float
        Convergence threshold for MMD^2
    """

    def __init__(
        self,
        env: GridEnvironment,
        mu_demonstrator: tuple[float, float],
        config_agent: dict,
        agent_name: str,
        solver: MDP_solver.MDPSolver,
        expert_features: np.ndarray,
        kernel_bandwidth: float = None,
        tol_mmd: float = 0.01,
    ):
        super().__init__(
            env=env,
            mu_demonstrator=mu_demonstrator,
            config_agent=config_agent,
            agent_name=agent_name,
            solver=solver,
        )
        self.expert_features = jnp.array(expert_features, dtype=jnp.float32)
        self.tol_mmd = tol_mmd
        self.kernel_bandwidth = kernel_bandwidth

    def compute_witness_reward_array(self, learner_features: np.ndarray) -> np.ndarray:
        """
        Compute witness function as per-state reward array.

        For each state s, compute:
          witness(s) = mean_i[k(phi(s), expert_i)] - mean_j[k(phi(s), learner_j)]

        where phi(s) is the feature vector from env.feature_matrix.

        Parameters
        ----------
        learner_features : ndarray, shape (m, d)

        Returns
        -------
        reward_array : ndarray, shape (n_states,)
        bandwidth : float
        """
        bandwidth = self.kernel_bandwidth
        if bandwidth is None:
            bandwidth = float(
                MMDLearner.median_heuristic(
                    self.expert_features, jnp.array(learner_features)
                )
            )

        expert_feat = np.array(self.expert_features)
        learner_feat = np.array(learner_features)
        bw = float(bandwidth)

        feature_matrix = self.env.get_state_feature_matrix()
        reward_array = np.zeros(self.env.n_states)

        for s in range(self.env.n_states):
            phi = feature_matrix[s].astype(np.float32)
            # k(phi, expert_i) for all i
            dists_expert = np.sum((phi - expert_feat) ** 2, axis=-1)
            k_expert = np.mean(np.exp(-dists_expert / (2 * bw**2)))
            # k(phi, learner_j) for all j
            dists_learner = np.sum((phi - learner_feat) ** 2, axis=-1)
            k_learner = np.mean(np.exp(-dists_learner / (2 * bw**2)))
            reward_array[s] = k_expert - k_learner

        return reward_array, bandwidth

    def _collect_trajectory_features(self, n_trajectories: int) -> np.ndarray:
        """
        Collect per-trajectory discounted feature sums.

        For each trajectory, compute sum of gamma^t * phi(s_t) using
        env.feature_matrix to map discrete state indices to feature vectors.

        Parameters
        ----------
        n_trajectories : int

        Returns
        -------
        features : ndarray, shape (n_trajectories, n_features)
        """
        feature_matrix = self.env.get_state_feature_matrix()
        feature_list = []

        for _ in range(n_trajectories):
            trajectory = self.solver.generate_episode(self.env, self.policy, self.T)
            if len(trajectory) == 0:
                feature_list.append(np.zeros(self.env.n_features, dtype=np.float32))
                continue

            feat_sum = feature_matrix[trajectory[0][0]].astype(np.float32)
            for i in range(len(trajectory)):
                feat_sum += (
                    self.env.gamma ** (i + 1)
                    * feature_matrix[trajectory[i][2]].astype(np.float32)
                )
            feature_list.append(feat_sum)

        return np.array(feature_list)

    def batch_MCE(
        self, early_stop_window: int = 200, **kwargs
    ) -> tuple[int, list[float]]:
        """
        Modified batch MCE using MMD witness function as reward.

        Key differences from standard batch_MCE:
        - No theta_e or theta_v parameters to optimize
        - Reward = witness function evaluated per state
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
        learner_features = None
        t = 1
        mmd_history = []
        n_traj = self.n_trajectories or 100

        if isinstance(self.env, ObjectWorldEnvironment):
            self.rewards = []

        while True:
            start = time()

            # Compute reward from MMD witness function
            if learner_features is None:
                reward_array = np.zeros(self.env.n_states)
                bw = self.kernel_bandwidth or 1.0
            else:
                reward_array, bw = self.compute_witness_reward_array(learner_features)

            # Solve for policy using exact solver
            variance_array = np.zeros(self.env.n_states)
            self.policy = self.solver.soft_value_iteration(
                self.env,
                dict(reward=reward_array, variance=variance_array),
            )

            if isinstance(self.env, ObjectWorldEnvironment):
                self.rewards.append(reward_array)

            # Collect per-trajectory feature sums
            learner_features = self._collect_trajectory_features(n_traj)

            # Compute MMD^2
            if self.kernel_bandwidth is None:
                bw = float(
                    MMDLearner.median_heuristic(
                        self.expert_features, jnp.array(learner_features)
                    )
                )

            mmd_sq = float(
                MMDLearner.compute_mmd_squared(
                    self.expert_features, jnp.array(learner_features), bw
                )
            )

            end = time()
            runtime.append(end - start)
            mmd_history.append(mmd_sq)

            wandb.log(
                {
                    f"mmd_squared_{self.agent_name}": mmd_sq,
                    f"step_{self.agent_name}": t,
                    f"kernel_bandwidth_{self.agent_name}": float(bw),
                }
            )

            # Convergence check
            if mmd_sq < self.tol_mmd and t >= self.miniter:
                break
            if t >= self.maxiter:
                break

            # Early stopping: check if MMD^2 is stagnating
            if t >= self.miniter and t >= early_stop_window:
                if self._check_stagnation(mmd_history, early_stop_window):
                    wandb.log({f"early_stopped_{self.agent_name}": True})
                    break

            t += 1

        # Fit linear reward for interpretability/comparison
        self.theta_e = self._fit_linear_reward(learner_features, bw)
        self.reward = self.env.get_reward_for_given_theta(self.theta_e)

        return t, runtime

    def _fit_linear_reward(self, learner_features: np.ndarray, bandwidth: float):
        """
        Fit theta such that r(s) ~ theta * phi(s) approximates the witness function.

        Parameters
        ----------
        learner_features : ndarray
        bandwidth : float

        Returns
        -------
        theta : ndarray
        """
        all_features = np.concatenate(
            [np.array(self.expert_features), np.array(learner_features)], axis=0
        )

        # Compute witness values for all feature vectors
        expert_feat = np.array(self.expert_features)
        learner_feat = np.array(learner_features)
        bw = float(bandwidth)

        witness_values = np.zeros(len(all_features))
        for i, phi in enumerate(all_features):
            phi = phi.astype(np.float32)
            dists_expert = np.sum((phi - expert_feat) ** 2, axis=-1)
            k_expert = np.mean(np.exp(-dists_expert / (2 * bw**2)))
            dists_learner = np.sum((phi - learner_feat) ** 2, axis=-1)
            k_learner = np.mean(np.exp(-dists_learner / (2 * bw**2)))
            witness_values[i] = k_expert - k_learner

        theta, _, _, _ = np.linalg.lstsq(all_features, witness_values, rcond=None)
        return theta
