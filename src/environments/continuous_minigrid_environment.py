"""Continuous version of the minigrid environment wrapper, without special feature extractors. This will be used to train Deep RL agents on different minigrid environments."""

from environments.environment import ContinuousEnvironment
from policy import Policy
import minigrid
import numpy as np

from pathlib import Path
import gymnasium as gym
import imageio


class ContinuousMinigridEnvironment(ContinuousEnvironment):
    """Wrapper class for the mingrid environments <https://minigrid.farama.org/> in order to work with our agents

    Parameters
    ----------
    env_args: dict[Any]
        environment definition parameters depending on the specific environment used
    """

    def __init__(self, env_args: dict):

        super().__init__(env_args)

        self.env = gym.make(
            env_args["env_name"],
            render_mode=env_args["render_mode"],
            size=env_args["grid_size"],
            max_steps=env_args["T"],
        )

        self.env = minigrid.wrappers.ImgObsWrapper(self.env)

        self.env_name = env_args["env_name"]

        self.seed = env_args["seed"]

        self._base_env = self.env
        self.env_val = self.env

        self.n_features = self.env.observation_space.sample().flatten().shape[0]

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
        dir = Path("recordings") / "cont_minigrid" / self.env_name
        dir.mkdir(parents=True, exist_ok=True)
        images = []
        terminated = False
        truncated = False
        state = self.reset()[0]
        img = self.env.render()
        images.append(img)
        t = 0
        while (not (terminated or truncated)) and t < T:
            # Take the action (index) that have the maximum expected future reward given that state
            action = policy.predict(state, t)
            state, _, terminated, truncated, _ = self.step(action)
            img = self.env.render()
            images.append(img)
            t += 1
        if store:
            imageio.mimsave(
                dir / f"{strname}.mp4",
                [np.array(img) for i, img in enumerate(images)],
                fps=fps,
            )

    def set_custom_reward_function(self, custom_reward_fn):
        self.env = ObsBasedRewardWrapper(self._base_env, custom_reward_fn)


class ObsBasedRewardWrapper(gym.Wrapper):
    def __init__(self, env, custom_reward_fn):
        super().__init__(env)
        self.custom_reward_fn = custom_reward_fn

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        custom_reward = self.custom_reward_fn(obs)

        return obs, custom_reward, terminated, truncated, info
