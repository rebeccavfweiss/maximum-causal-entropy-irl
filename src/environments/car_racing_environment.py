from environments.environment import Environment
from policy import Policy
from stable_baselines3.common.vec_env import (
    VecFrameStack,
    VecVideoRecorder,
    VecEnvWrapper,
    VecEnv
)
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import os
import numpy as np


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
        
        self._base_env = self.env

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
            true reward for the given policy (approximated based on trajectories)
        """
        mean_reward, std_reward = evaluate_policy(agent.policy.model, self.env, n_eval_episodes=n_trajectories, return_episode_rewards=True)
        
        # TODO remove this once everything works as expected this was only in order to test whether or not overwriting the reward function works
        #mean_reward, std_reward = self.evaluate_policy_custom(agent.policy.model, self.env, n_trajectories)
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        return mean_reward
    
    def set_custom_reward_function(self, custom_reward_fn):
        """Wrap the original vec_env with a custom reward function."""
        self.env = VecCustomRewardWrapper(self._base_env, custom_reward_fn)

    def reset_reward_function(self):
        """Reset to the original vec_env with default rewards."""
        self.env = self._base_env

    def evaluate_policy_custom(
        self,
        model,
        vec_env: VecEnv,
        n_eval_episodes: int = 5,
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
            total_reward = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = vec_env.step(action)

                total_reward += reward[0]  # reward is vectorized: shape (n_envs,)

                if render:
                    vec_env.render()

            episode_rewards.append(total_reward)

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        return mean_reward, std_reward


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
        custom_rewards = np.array([
            self.custom_reward_fn(next_obs[i])
            for i in range(len(rewards))
        ])
        return next_obs, custom_rewards, dones, info