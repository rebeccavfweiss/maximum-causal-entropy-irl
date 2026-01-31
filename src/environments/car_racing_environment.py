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
import gym


class CarRacingEnvironment(ContinuousEnvironment):
    """
    Wrapper class for the gymnasium car racing environment <https://gymnasium.farama.org/environments/box2d/car_racing/>
    in order to work with our agents

    Parameters
    ----------
    env_args: dict[Any]
        environment definition parameters depending on the specific environment used

    """

    def __init__(self, env_args: dict):

        super().__init__(env_args)

        self.frame_width = env_args["width"]
        self.frame_height = env_args["height"]
        self.n_frames = env_args["n_frames"]
        self.lap_complete_percent = env_args["lap_complete_percent"]

        self.continuous_actions = env_args["continuous_actions"]

        env = VecFrameStack(self.make_env(), n_stack=env_args["n_frames"])
        self.env = VecTransposeImage(env)

        env = VecFrameStack(self.make_env(), n_stack=env_args["n_frames"])
        self.env_val = VecTransposeImage(env)

        self._base_env = self.env

        self.n_features = self.env.observation_space.sample().flatten().shape[0]

    def make_env(self):
        env_kwargs = {
            "render_mode": "rgb_array",
            "continuous": self.continuous_actions,
            "lap_complete_percent": self.lap_complete_percent,
            "domain_randomize": False,
            "max_episode_steps": self.T,
        }
        env = make_vec_env(
            "CarRacing-v3",
            n_envs=1,
            wrapper_class=WarpFrame,
            wrapper_kwargs={"width": self.frame_width, "height": self.frame_height},
            env_kwargs=env_kwargs,
        )
        # env = VecNormalizeObs(env, NormalizeObs)

        return env

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

        name_prefix = "car_racing" + (
            "_continuous" if self.continuous_actions else "_discrete"
        )
        env = VecVideoRecorder(
            self.env,
            video_folder=os.path.dirname(f"recordings\car_racing\{strname}.mp4") or ".",
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
            / "car_racing"
            / f"{name_prefix}_{strname}-step-0-to-step-{T}.mp4"
        )

    def set_custom_reward_function(self, custom_reward_fn):
        """Wrap the original vec_env with a custom reward function."""
        self.env = VecCustomRewardWrapper(self._base_env, custom_reward_fn)

    def set_max_episode_steps(self, new_steps: int):
        for env in self.env.envs:
            base_env = env
            while isinstance(base_env, gym.Wrapper):
                if isinstance(base_env, gym.wrappers.TimeLimit):
                    base_env._max_episode_steps = new_steps
                    break
                base_env = base_env.env


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


class NormalizeObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )

    def observation(self, obs):
        return obs.astype(np.float32) / 255.0


class VecNormalizeObs(VecEnvWrapper):
    def __init__(self, venv, obs_wrapper_cls):
        self.obs_wrapper = obs_wrapper_cls(venv.envs[0])
        super().__init__(venv)

    def reset(self):
        obs = self.venv.reset()
        return obs.astype(np.float32) / 255.0

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return obs.astype(np.float32) / 255.0, rewards, dones, infos
