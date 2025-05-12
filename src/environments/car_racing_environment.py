from environments.environment import Environment
from policy import Policy
import gymnasium as gym
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecVideoRecorder,
)
from stable_baselines3.common.env_util import make_vec_env
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

        env = DummyVecEnv([self.make_env])
        self.env = VecFrameStack(env, n_stack=env_args["n_frames"])
        self.n_actions = self.env.action_space.n

        self.n_states = (
            self.frame_height * self.frame_width * self.n_colors * self.n_colors + 1
        )

    def make_env(self):
        env = gym.make(
            "CarRacing-v3",
            render_mode="rgb_array",
            continuous=False,
            lap_complete_percent=self.lap_complete_percent,
            domain_randomize=False,
        )
        # reduce size of the observations
        env = KMeansResizeWrapper(
            env,
            width=self.frame_width,
            height=self.frame_height,
            n_clusters=self.n_colors,
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


class KMeansResizeWrapper(gym.ObservationWrapper):
    """
    Environment wrapper in order to reduce the size of the observation space

    Parameters
    ----------
    env : gym.Env
        environment on which the wrapper should be applied
    width : int
        new width of the frame
    height : int
        new height of the frame
    n_clusters : int
        number of color labels to use
    """

    def __init__(
        self, env: gym.Env, width: int = 84, height: int = 84, n_clusters: int = 6
    ):
        super().__init__(env)
        self.width = width
        self.height = height
        self.n_clusters = n_clusters

        # Updated observation space: single channel (cluster index per pixel)
        self.observation_space = Box(
            low=0,
            high=n_clusters - 1,
            shape=(self.height, self.width, 1),
            dtype=np.uint8,
        )

        # Pre-train KMeans centroids on a few random frames
        self.kmeans = self._init_kmeans()

    def _init_kmeans(self):
        print("Initializing KMeans clusters on sample frames")
        samples = []

        for _ in range(10):
            obs, _ = self.env.reset()
            img = cv2.resize(
                obs, (self.width, self.height), interpolation=cv2.INTER_AREA
            )
            flat_pixels = img.reshape(-1, 3)
            idx = np.random.choice(flat_pixels.shape[0], 1000, replace=False)
            samples.append(flat_pixels[idx])

        all_samples = np.concatenate(samples, axis=0)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init="auto").fit(
            all_samples
        )
        return kmeans

    def observation(self, obs):
        # Resize
        resized = cv2.resize(
            obs, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        # Flatten and apply KMeans
        flat = resized.reshape(-1, 3)
        labels = self.kmeans.predict(flat)
        clustered = labels.reshape(self.height, self.width, 1).astype(np.uint8)
        return clustered
