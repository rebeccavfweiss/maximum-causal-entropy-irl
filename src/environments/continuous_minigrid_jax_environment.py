"""
Continuous MiniGrid environment wrapper for JAX-based training.

Unlike ContinuousMinigridEnvironment, does NOT use SB3 wrappers.
Uses raw Gymnasium API, compatible with JaxDQNTrainer and JaxSACTrainer.
"""

import numpy as np
import gymnasium as gym
import minigrid
from pathlib import Path
import imageio
from environments.environment import ContinuousEnvironment
from policy import Policy


class ContinuousMinigridJaxEnvironment(ContinuousEnvironment):
    """
    MiniGrid wrapper for JAX-based training.

    Uses raw Gymnasium API with ImgObsWrapper, compatible with
    JaxDQNTrainer/JaxSACTrainer (including CNN networks for image obs).

    Parameters
    ----------
    env_args : dict
        Must include:
        - env_name: str (e.g., "MiniGrid-DoorKey-5x5-v0")
        - grid_size: int
        - T: int (max episode steps)
        - gamma: float
        Optional:
        - render_mode: str (default "rgb_array")
        - seed: int or None
    """

    def __init__(self, env_args: dict):
        super().__init__(env_args)

        self._gym_env = gym.make(
            env_args["env_name"],
            render_mode=env_args.get("render_mode", "rgb_array"),
            size=env_args["grid_size"],
            max_steps=env_args["T"],
        )
        self._gym_env = minigrid.wrappers.ImgObsWrapper(self._gym_env)

        self.env = self._gym_env
        self._base_env = self._gym_env
        self.env_name = env_args["env_name"]
        self.seed = env_args.get("seed")

        self.n_features = int(np.prod(self._gym_env.observation_space.shape))
        self.n_actions = self._gym_env.action_space.n
        self._custom_reward_fn = None

    def get_gym_env(self) -> gym.Env:
        """Returns the raw gymnasium env for direct interaction by JAX trainers."""
        return self._gym_env

    def reset(self):
        """Reset wrapper compatible with Gymnasium API."""
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
        """Record a video of the given policy in the environment."""
        dir = Path("recordings") / "cont_minigrid_jax" / self.env_name
        dir.mkdir(parents=True, exist_ok=True)
        images = []
        done = False
        state = self.reset()
        img = self._gym_env.render()
        images.append(img)
        t = 0
        while (not done) and t < T:
            action = policy.predict(state, t)
            state, _, done, _ = self.step(action)
            img = self._gym_env.render()
            images.append(img)
            t += 1
        if store:
            path = dir / f"{strname}.mp4"
            imageio.mimsave(
                path,
                [np.array(img) for img in images],
                fps=fps,
            )
            return path
        return None

    def compute_true_reward_for_agent(
        self, agent, n_trajectories: int = None, T: int = None
    ) -> float:
        """Evaluate using the solver's generate_episode."""
        rewards = []
        for _ in range(n_trajectories):
            trajectory = agent.solver.generate_episode(self, agent.policy, T)
            rewards.append(
                sum(
                    trajectory[j][3] * self.gamma ** j
                    for j in range(len(trajectory))
                )
            )
        return np.mean(rewards)
