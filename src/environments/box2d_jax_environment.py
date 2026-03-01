"""
Box2D environment wrapper for JAX-based training.

Unlike Box2DEnvironment, does NOT use SB3 VecNormalize/DummyVecEnv.
Uses raw Gymnasium API, compatible with JaxDQNTrainer and JaxSACTrainer.
"""

import numpy as np
import gymnasium as gym
from pathlib import Path
from environments.environment import ContinuousEnvironment
from policy import Policy


class Box2DJaxEnvironment(ContinuousEnvironment):
    """
    LunarLander/BipedalWalker wrapper for JAX-based training.

    Unlike Box2DEnvironment, does NOT use SB3 VecNormalize/DummyVecEnv.
    Uses raw Gymnasium API directly, compatible with JaxDQNTrainer/JaxSACTrainer.

    Parameters
    ----------
    env_args : dict
        Must include:
        - env_id: str (e.g., "LunarLander-v3", "LunarLanderContinuous-v3")
        - gamma: float
        - T: int
        - continuous: bool
    """

    def __init__(self, env_args: dict):
        super().__init__(env_args)
        self.env_id = env_args["env_id"]
        self.continuous = env_args.get("continuous", False)

        # Raw gymnasium environment (no SB3 wrappers)
        self._gym_env = gym.make(self.env_id)
        self.env = self._gym_env
        self._base_env = self._gym_env

        self.n_features = self._gym_env.observation_space.shape[0]
        if self.continuous:
            self.n_actions = self._gym_env.action_space.shape[0]
        else:
            self.n_actions = self._gym_env.action_space.n

        self._custom_reward_fn = None

    def get_gym_env(self) -> gym.Env:
        """Returns the raw gymnasium env for direct interaction by JAX trainers."""
        return self._gym_env

    def reset(self):
        """Reset wrapper compatible with both Gymnasium API versions."""
        obs, _ = self._gym_env.reset()
        return obs

    def step(self, action):
        """Step wrapper that applies custom reward function if set."""
        obs, reward, terminated, truncated, info = self._gym_env.step(action)
        if self._custom_reward_fn is not None:
            reward = self._custom_reward_fn(obs)
        return obs, reward, terminated or truncated, info

    def set_custom_reward_function(self, custom_reward_fn):
        """Set a custom reward function r(obs) -> float."""
        self._custom_reward_fn = custom_reward_fn

    def reset_reward_function(self):
        """Reset to the original environment rewards."""
        self._custom_reward_fn = None

    def render(
        self,
        policy: Policy,
        T: int = 20,
        store: bool = False,
        strname: str = "",
        fps: int = 1,
        **kwargs,
    ) -> Path:
        """
        Record a video of the given policy in the environment.

        Parameters
        ----------
        policy : Policy
            Policy to use
        T : int
            Maximal episode length
        store : bool
            Whether to store the rendering
        strname : str
            File name to store
        fps : int
            Frames per second

        Returns
        -------
        path : Path or None
            Path to the video file, or None if not stored
        """
        if not store:
            return None

        render_env = gym.make(self.env_id, render_mode="rgb_array")
        video_dir = Path("recordings") / self.env_id
        video_dir.mkdir(parents=True, exist_ok=True)

        render_env = gym.wrappers.RecordVideo(
            render_env,
            str(video_dir),
            name_prefix=f"{self.env_id}_{strname}",
            episode_trigger=lambda ep: ep == 0,
        )

        obs, _ = render_env.reset()
        total_reward = 0.0
        step = 0

        for step in range(T):
            action = policy.predict(obs, step)
            obs, reward, terminated, truncated, _ = render_env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        print(f"Episode done after: {step + 1} steps with reward={total_reward}")
        render_env.close()

        return None  # Video path depends on gymnasium's naming convention

    def compute_true_reward_for_agent(
        self, agent, n_trajectories: int = None, T: int = None
    ) -> float:
        """Evaluate using the solver's generate_episode."""
        rewards = []
        for _ in range(n_trajectories):
            trajectory = agent.solver.generate_episode(self, agent.policy, T)
            rewards.append(
                sum(
                    trajectory[j][3] * self.gamma**j
                    for j in range(len(trajectory))
                )
            )
        return np.mean(rewards)
