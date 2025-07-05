import numpy as np
from pathlib import Path
from MDP_solver import MDPSolver
from environments.environment import Environment
from policy import Policy, ModelPolicy
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from utils import TimedEvalCallback
from wandb.integration.sb3 import WandbCallback

import os
os.environ["WANDB_DISABLE_SYMLINK"] = "true"


np.set_printoptions(suppress=True)
np.set_printoptions(precision=12)
np.set_printoptions(linewidth=500)


class MDPSolverApproximation(MDPSolver):
    """
    abstract class collecting the basic methods a MDP solver needs
    For this solver we expect the environment to be large w.r.t the observation space
    If the environment is, e.g., an easy gridworld with a known enumeration of the states then use MDPSolverExact instead as it will be more accurate.

    Parameters
    ----------
    T : int
        finite horizon value
    compute_variance : bool
            whether or not variance term should be computed (for efficiency reasons will only be computed if necessary)
    sac_timesteps : int
        timesteps for embedded SAC training in the approximated SVI
    sac_buffer_size : int
        buffer size for embedded SAC training in the approximated SVI
    sac_tau : float
        soft update coefficient ("Polyak update", between 0 and 1)
    log_dir : str
        logging directory
    model_dir : str
        local model storage directory
    """

    def __init__(
        self,
        T: int,
        compute_variance: bool,
        sac_timesteps: int = 10000,
        sac_buffer_size: int = 100000,
        sac_tau: float = 0.005,
        log_dir: str = None,
        model_dir: str = None,
    ):

        super().__init__(T, compute_variance)

        self.sac_timesteps = sac_timesteps
        self.sac_buffer_size = sac_buffer_size
        self.sac_tau = sac_tau

        self.log_dir = log_dir
        self.model_dir = model_dir

    def compute_feature_SVF_bellmann(
        self,
        env: Environment,
        policy: Policy,
        trajectory: list[tuple[int, int, int, float]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        computes feature SVF

        Parameters
        ----------
        env : environment.Environment
            the environment representing the setting of the problem
        policy : ndarray
            not used here but given for uniform interface
        trajectory : list[tuple[any,int,int, float]]
            predefined trajectory of an agent

        Returns
        -------
        feature_expectation : ndarray
        feature_variance : ndarray
        """
        feature_sum = trajectory[0][0].flatten() + sum(
            env.gamma ** (t + 1) * trajectory[t][2].flatten()
            for t in range(len(trajectory))
        )

        feature_sum = feature_sum.astype(np.float32)

        if self.compute_variance:
            feature_sum_prod = np.outer(feature_sum, feature_sum)
        else:
            feature_sum_prod = np.zeros(
                (feature_sum.shape[0], feature_sum.shape[0]), dtype=np.float32
            )

        return feature_sum, feature_sum_prod


class MDPSolverApproximationExpectation(MDPSolverApproximation):
    """
    MDP solver that uses feature expectation matching using an approximation for Soft Value Iteration by using a Soft Critic Actor algorithm

    Parameters
    ----------
    experiment_name : str
        name for the experiment that will be used in naming logging/storage directories
    T : int
        finite horizon value
    compute_variance : bool
            whether or not variance term should be computed (for efficiency reasons will only be computed if necessary)
    sac_timesteps : int
        timesteps for embedded SAC training in the approximated SVI
    sac_buffer_size : int
        buffer size for embedded SAC training in the approximated SVI
    sac_tau : float
        soft update coefficient ("Polyak update", between 0 and 1)
    """

    def __init__(
        self,
        experiment_name: str,
        T: int = 45,
        compute_variance: bool = False,
        sac_timesteps: int = 10000,
        sac_buffer_size: int = 100000,
        sac_tau: float = 0.005
    ):
        super().__init__(
            T,
            compute_variance,
            sac_timesteps,
            sac_buffer_size,
            sac_tau,
            log_dir=Path("experiments")/experiment_name/"agent_expectation",
            model_dir=Path("models")/experiment_name/"agent_expectation",
        )

    def soft_value_iteration(self, env: Environment, values: dict[str:any]) -> Policy:
        """
        computes soft value iteration using feature expectation matching (using recurive evaluation as finite horizon)

        Parameters
        ----------
        env : environment.Environment
            the environment representing the setting of the problem
        values : dict[str:any]
            dictionary with the feature expecation and variance functions

        Returns
        -------
        pi_s : Policy
            policy based on the learned model
        """

        env.set_custom_reward_function(lambda s: values["reward"](s.flatten()))

        model = SAC("CnnPolicy", env.env, verbose=0, buffer_size=self.sac_buffer_size, gamma=env.gamma, tau=self.sac_tau)

        callback = CallbackList(
            [
                TimedEvalCallback(
                    env.env,
                    best_model_save_path=self.model_dir,
                    log_path=self.log_dir,
                    eval_freq=max(10, int(self.sac_timesteps / 100)),
                    render=False,
                    n_eval_episodes=5,
                ),
                WandbCallback(model_save_path=self.model_dir, verbose=1),
            ]
        )

        model.learn(
            total_timesteps=self.sac_timesteps, callback=callback, progress_bar=True
        )

        env.reset_reward_function()

        model = SAC.load(self.model_dir/"best_model")

        # actually return best trained model and not last
        return ModelPolicy(SAC.load(self.model_dir/"best_model"))


class MDPSolverApproximationVariance(MDPSolverApproximation):
    """
    MDP solver that uses feature expectation and variance matching using an approximation for Soft Value Iteration by using a Soft Critic Actor algorithm

    Parameters
    ----------
    experiment_name : str
        name for the experiment that will be used in naming logging/storage directories
    T : int
        finite horizon value
    compute_variance : bool
            whether or not variance term should be computed (for efficiency reasons will only be computed if necessary)
    sac_timesteps : int
        timesteps for embedded SAC training in the approximated SVI
    sac_buffer_size : int
        buffer size for embedded SAC training in the approximated SVI
    sac_tau : float
        soft update coefficient ("Polyak update", between 0 and 1)
    """

    def __init__(
        self,
        experiment_name: str,
        T: int = 45,
        compute_variance: bool = True,
        sac_timesteps: int = 10000,
        sac_buffer_size: int = 100000,
        sac_tau : float = 0.005
    ):
        super().__init__(
            T,
            compute_variance,
            sac_timesteps,
            sac_buffer_size,
            sac_tau,
            log_dir=Path("experiments")/experiment_name/"agent_variance",
            model_dir=Path("models")/experiment_name/"agent_variance",
        )

    def soft_value_iteration(self, env: Environment, values: dict[str:any]) -> Policy:
        """
        computes soft value iteration using feature expectation matching (using recurive evaluation as finite horizon)

        Parameters
        ----------
        env : environment.Environment
            the environment representing the setting of the problem
        values : dict[str:any]
            dictionary with the feature expecation and variance term needed for bellmann

        Returns
        -------
        pi_s : Policy
            learned policy
        """

        # Assumption: gamma = 1 in order to simplify reward term
        env.set_custom_reward_function(
            lambda s: values["reward"](s.flatten()) + values["variance"](s.flatten())
        )

        model = SAC("CnnPolicy", env.env, verbose=0, buffer_size=self.sac_buffer_size, gamma=env.gamma, tau = self.sac_tau)

        callback = CallbackList(
            [
                TimedEvalCallback(
                    env.env,
                    best_model_save_path=self.model_dir,
                    log_path=self.log_dir,
                    eval_freq=max(10, int(self.sac_timesteps / 100)),
                    render=False,
                    n_eval_episodes=5,
                ),
                WandbCallback(model_save_path=self.model_dir, verbose=1),
            ]
        )

        model.learn(
            total_timesteps=self.sac_timesteps, callback=callback, progress_bar=True
        )

        env.reset_reward_function()

        # actually return best trained model and not last
        return ModelPolicy(SAC.load(self.model_dir / "best_model"))
