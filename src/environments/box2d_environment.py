from environments.environment import ContinuousEnvironment
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import numpy as np

from environments.environment import ContinuousEnvironment
from policy import Policy
from stable_baselines3.common.vec_env import (
    VecFrameStack,
    VecVideoRecorder,
    VecEnvWrapper,
)
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.env_util import make_vec_env
import os
import numpy as np
import utils
from pathlib import Path
import gymnasium as gym


class Box2DEnvironment(ContinuousEnvironment):
    """
    Wrapper for LunarLander-v2 and BipedalWalker-v3 environments.
    """

    def __init__(self, env_args: dict):
        super().__init__(env_args)
        self.env_id = env_args["env_id"]  # e.g., "LunarLanderContinuous-v2"
        self.n_envs = env_args.get("n_envs", 1)

        # Initialize vectorized environment
        env = make_vec_env(
            self.env_id,
            n_envs=self.n_envs,
            wrapper_class=None,  # Add custom wrappers here if needed
        )
        self.continuous = env_args["continuous"]

        # Apply normalization (standard practice for vector-based Box2D envs)
        self.env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        self.env_val = VecNormalize(
            env, norm_obs=True, norm_reward=False, clip_obs=10.0
        )
        self._base_env = self.env

        self.n_features = self.env.observation_space.sample().shape[0]

    def compute_true_reward_for_agent(self, agent, n_trajectories, T):
        """Calculates the mean reward for an agent over multiple episodes."""
        total_rewards = []
        for _ in range(n_trajectories):
            obs = self.env.reset()
            episode_reward = 0
            for _ in range(T):
                action = agent.policy.predict(obs)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward[0]
                if done[0]:
                    break
            total_rewards.append(episode_reward)
        return np.mean(total_rewards)

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
        Function to record a video of the given policy in the environment

        Parameters
        ----------
        pi : ndarray
            policy to use
        T : int
            maximal episode length
        store : bool
            whether to store the rendering
        strname : str
            file name to store
        fps : int
            frames per second

        Returns
        -------
        path : Path
            path to the file in which the video is stored
        """

        name_prefix = self.env_id + ("_continuous" if self.continuous else "_discrete")
        env = VecVideoRecorder(
            self.env,
            video_folder=os.path.dirname(f"recordings\{self.env_id}\{strname}.mp4")
            or ".",
            record_video_trigger=lambda step: True,  # record first episode
            video_length=T,
            name_prefix=f"{name_prefix}_{strname}",
        )

        obs = env.reset()
        terminated = False
        info = []
        step = 0
        total_reward = 0

        while (not (terminated or utils.is_truncated_from_infos(info))) and (step < T):
            action = policy.predict(obs, step)
            obs, reward, terminated, info = env.step(action)
            step += 1
            total_reward += reward

        print("Episode done after: ", step, " steps with reward=", total_reward)

        env.close()

        return (
            Path("recordings")
            / self.env_id
            / f"{name_prefix}_{strname}-step-0-to-step-{T}.mp4"
        )

    def set_custom_reward_function(self, custom_reward_fn):
        """Wrap the original vec_env with a custom reward function."""
        self.env = VecCustomRewardWrapper(self._base_env, custom_reward_fn)


class VecCustomRewardWrapper(VecEnvWrapper):
    def __init__(self, venv, custom_reward_fn):
        """
        :param venv: Vectorized environment
        :param custom_reward_fn: function(next_obs, original_reward, info) -> custom_reward
        """
        super().__init__(venv)
        self.custom_reward_fn = custom_reward_fn

    def reset(self):
        return self.venv.reset()

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        next_obs, rewards, dones, info = self.venv.step_wait()
        # Apply custom reward function vector-wise
        custom_rewards = np.array(
            [self.custom_reward_fn(next_obs[i]) for i in range(len(rewards))]
        )
        return next_obs, custom_rewards, dones, info
