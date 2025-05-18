from environments.environment import Environment
from policy import Policy
import gymnasium as gym
from stable_baselines3.common.vec_env import (
    VecFrameStack,
    VecVideoRecorder,
)
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
from gymnasium.spaces import Box


class CarRacingEnvironment(Environment):
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
        self.n_colors = env_args["n_colors"]
        self.n_frames = env_args["n_frames"]
        self.lap_complete_percent = env_args["lap_complete_percent"]

        self.log_dir = "./experiments"

        env = VecFrameStack(self.make_env(), n_stack=env_args["n_frames"])
        self.env = VecTransposeImage(env)

        env = VecFrameStack(self.make_env(), n_stack=env_args["n_frames"])
        self.env_val = VecTransposeImage(env)

        self.eval_callback = EvalCallback(self.env_val,
                             best_model_save_path=self.log_dir,
                             log_path=self.log_dir,
                             eval_freq=25_000,
                             render=False,
                             n_eval_episodes=5)

    def make_env(self):
        env_kwargs = {
            "render_mode": "rgb_array",
            "continuous": False,
            "lap_complete_percent": self.lap_complete_percent,
            "domain_randomize": False
        }

        env = make_vec_env(
            "CarRacing-v3",
            n_envs=1,
            wrapper_class=WarpFrame,
            env_kwargs=env_kwargs
        )

        return env

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

    def render(
        self,
        policy: Policy,
        T: int = 20,
        store: bool = False,
        strname: str = "",
        fps: int = 1,
        **kwargs,
    ) -> None:
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
        """

        env = VecVideoRecorder(
            self.env,
            video_folder=os.path.dirname(f"recordings\car_racing\{strname}.mp4") or ".",
            record_video_trigger=lambda step: True,  # record first episode
            video_length=T,
            name_prefix="car_racing",
        )

        obs = env.reset()
        terminated = False
        step = 0
        total_reward = 0

        while (not terminated) and (step < T):
            action = policy.predict(obs, step)
            obs, reward, terminated, _ = env.step(action)
            step += 1
            total_reward += reward

        print("Episode done after: ", step, " steps with reward=", total_reward)

        env.close()

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

        mean_reward, std_reward = evaluate_policy(agent.pi.model, self.env, n_eval_episodes=n_trajectories)
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        return mean_reward