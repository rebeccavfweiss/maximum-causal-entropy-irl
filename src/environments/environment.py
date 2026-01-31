import numpy as np
from abc import ABC, abstractmethod
from scipy import sparse
from policy import Policy
from pathlib import Path
import utils
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gym

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)


class Environment(ABC):
    """
    Base class to generalize access to different types of environments

    Parameters
    ----------
    env_args: dict[Any]
        environment definition parameters depending on the specific environment used
    """

    def __init__(self, env_args: dict):

        self.gamma = env_args["gamma"]
        self.terminal_states = None
        self.reward = None

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def render(self, policy: Policy, T: int = 20, store: bool = False, **kwargs):
        pass

    def compute_true_reward_for_agent(
        self, agent, n_trajectories: int = 1, T: int = None
    ) -> float:
        """
        Compute the true reward in the environment either with the given policy or with trajectories

        Parameters
        ----------
        agent
            agent/demonstrator that should be evaluated in the environment
        n_trajectories : int
            number of trajectories to use

        Returns
        -------
        reward : float
            true reward for the given agent
        """

        # compute reward based on trajectories
        rewards = []
        for i in range(n_trajectories):
            trajectory = agent.solver.generate_episode(self, agent.policy, T)
            rewards.append(
                sum([trajectory[j][3] * self.gamma**j for j in range(len(trajectory))])
            )
        return np.mean(rewards)


class GridEnvironment(Environment):
    """
    Base class to generalize access to SimpleEnvironment and Gymnasium environments

    Parameters
    ----------
    env_args: dict[Any]
        environment definition parameters depending on the specific environment used
    """

    def __init__(self, env_args: dict):

        super().__init__(env_args)
        self.theta_reward = env_args["theta"]

        # true reward per state
        self.reward = None

    def get_reward_for_given_theta(self, theta_e: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        theta_e : ndarray

        Returns
        -------
        reward : ndarray
        """
        reward = self.feature_matrix.dot(theta_e)
        return np.array(reward)

    def get_variance_for_given_theta(self, theta_v: np.ndarray) -> np.ndarray:
        """
        computes the variance term for soft value iteration for given theta_v

        Parameters
        ----------
        theta_v : ndarray

        Returns
        -------
        variance : ndarray
        """
        variance = [
            (theta_v.dot(self.feature_matrix[i, :])).dot(self.feature_matrix[i, :])
            for i in range(self.feature_matrix.shape[0])
        ]

        return np.array(variance)

    def get_state_feature_matrix(self) -> np.ndarray:
        """
        Getter function for (state) feature matrix
        """
        return self.feature_matrix.copy()

    @abstractmethod
    def _compute_state_feature_matrix(self):
        pass

    @abstractmethod
    def _compute_transition_matrix(self):
        pass

    @abstractmethod
    def _get_initial_distribution(self):
        pass

    def _compute_transition_sparse_list(self) -> list[sparse.csc_matrix]:
        """
        Computes the sparse transition list for the different actions

        Returns
        -------
        T_sparse_list : list[sparse.csr_matrix]
        """
        T_sparse_list = []
        for a in range(self.n_actions):
            T_sparse_list.append(sparse.csr_matrix(self.T_matrix[:, :, a]))

        return T_sparse_list

    def compute_true_reward_for_agent(
        self, agent, n_trajectories: int = None, T: int = None
    ) -> float:
        """
        Compute the true reward in the environment either with the given policy or with trajectories

        Parameters
        ----------
        agent
            agent/demonstrator that should be evaluated in the environment
        n_trajectories : int
            number of trajectories to use, if None then we use the reward vector

        Returns
        -------
        reward : float
            true reward for the given policy
        """

        if n_trajectories is None or n_trajectories == 0:
            # use policy and static rewards directly

            return np.dot(
                self.reward, np.sum(agent.solver.compute_SV(self, agent.policy), axis=0)
            )

        else:
            # compute reward based on trajectories
            return super().compute_true_reward_for_agent(agent, n_trajectories, T)


class ContinuousEnvironment(Environment):
    """General wrapper for continuous environments, like the Car Racing or Minigrid environment."""

    def __init__(self, env_args: dict):

        super().__init__(env_args)
        self.T = env_args["T"]

        self.log_dir = Path("experiments")
        self.gamma = env_args["gamma"]

    def reset(self) -> any:
        """
        Reset wrapper to generalize environment access over different environments

        Returns
        -------
        Initial state description
        """

        state = self.env.reset()

        return state

    def step(self, action: int) -> tuple[any, float, bool, bool]:
        """
        Step wrapper to generalize environment access over different environments

        Parameters
        ----------
        action : int
            action to take

        Returns
        -------
        new_state
            current state description
        reward : float
            reward for taken action
        terminated : bool
            if episode is terminated
        truncated : bool
            if episode was truncated
        """
        new_state, reward, terminated, truncated = self.env.step(action)

        return new_state, reward, terminated, truncated

    @abstractmethod
    def render(self, policy: Policy, T: int = 20, store: bool = False, **kwargs):
        pass

    def compute_true_reward_for_agent(
        self, agent, n_trajectories: int = None, T: int = None
    ) -> float:
        """
        Compute the true reward in the environment either with the given policy or with trajectories

        Parameters
        ----------
        agent
            agent/demonstrator that should be evaluated in the environment
        n_trajectories : int
            number of trajectories to use, if None then we use the reward vector

        Returns
        -------
        reward : float
            true reward for the given policy (approximated based on trajectories)
        """

        if isinstance(self.env, VecEnv):
            mean_reward, std_reward = self.evaluate_policy_custom(
                agent.policy.model, self.env, n_trajectories, T
            )

        else:
            mean_reward, std_reward = evaluate_policy(
                agent.policy.model, self.env, n_eval_episodes=n_trajectories
            )
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        return mean_reward

    def evaluate_policy_custom(
        self,
        model,
        vec_env: VecEnv,
        n_eval_episodes: int = 5,
        T: int = 1000,
        deterministic: bool = True,
        render: bool = False,
    ) -> tuple[float, float]:
        """
        Custom evaluation function that uses actual env.step() rewards
        instead of relying on info["episode"]["r"].

        Assumes n_envs = 1 (can be extended).
        """
        episode_rewards = []
        n_envs = vec_env.num_envs
        assert n_envs == 1, "This custom evaluator only supports n_envs=1 for now."

        for _ in range(n_eval_episodes):
            obs = vec_env.reset()
            done = False
            info = []
            total_reward = 0.0
            t = 1
            while (not (done or utils.is_truncated_from_infos(info))) and t <= T:
                action, _ = model.predict(obs, deterministic=deterministic)

                obs, reward, done, info = vec_env.step(action)

                total_reward += reward[0]  # reward is vectorized: shape (n_envs,)

                if render:
                    vec_env.render()
                t += 1

            episode_rewards.append(total_reward)

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        return mean_reward, std_reward

    @abstractmethod
    def set_custom_reward_function(self, custom_reward_fn):
        """Wrap the original vec_env with a custom reward function."""
        pass

    def reset_reward_function(self):
        """Reset to the original vec_env with default rewards."""
        self.env = self._base_env

    def set_max_episode_steps(self, new_steps: int):
        for env in self.env.envs:
            base_env = env
            while isinstance(base_env, gym.Wrapper):
                if isinstance(base_env, gym.wrappers.TimeLimit):
                    base_env._max_episode_steps = new_steps
                    break
                base_env = base_env.env
