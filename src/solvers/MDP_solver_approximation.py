import numpy as np
from pathlib import Path
from MDP_solver import MDPSolver
from environments.environment import Environment
from policy import Policy, ModelPolicy
from stable_baselines3 import SAC, DQN
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
    continuous_actions : bool
        whether the environment has a continuous or discrete action space
    policy_config : dict[str, any]
        dictionary containing parameters for SAC/DQN, e.g.,
        buffer_size : int
            buffer size for embedded SAC/DQN training in the approximated SVI
        tau : float
            soft update coefficient ("Polyak update", between 0 and 1)
        train_freq: int
            number of training steps after which the target network should be updated
    training_timesteps : int
            timesteps for embedded SAC/DQN training in the approximated SVI
    log_dir : str
        logging directory
    model_dir : str
        local model storage directory
    """

    def __init__(
        self,
        T: int,
        compute_variance: bool,
        continuous_actions: bool,
        policy_config: dict[str, any],
        training_timesteps : int = 10000,
        log_dir: str = None,
        model_dir: str = None,
    ):

        super().__init__(T, compute_variance)

        self.continuous_actions = continuous_actions
        self.policy_config = policy_config
        self.training_timesteps = training_timesteps

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
    continuous_actions : bool
        whether or not variance term should be computed (for efficiency reasons will only be computed if necessary)
    policy_config : dict[str, any]
        dictionary containing parameters for SAC/DQN, e.g.,
        buffer_size : int
            buffer size for embedded SAC/DQN training in the approximated SVI
        tau : float
            soft update coefficient ("Polyak update", between 0 and 1)
        train_freq: int
            number of training steps after which the target network should be updated
    training_timesteps : int
            timesteps for embedded SAC/DQN training in the approximated SVI
    T : int
        finite horizon value
    compute_variance : bool
            whether or not variance term should be computed (for efficiency reasons will only be computed if necessary)
    """

    def __init__(
        self,
        experiment_name: str,
        continuous_actions : bool,
        policy_config : dict[str, any],
        training_timesteps : int =10000,
        T: int = 45,
        compute_variance: bool = False,
        
    ):
        super().__init__(
            T,
            compute_variance,
            continuous_actions,
            policy_config,
            training_timesteps,
            log_dir=Path("experiments")/experiment_name/"agent_expectation",
            model_dir=Path("models")/experiment_name/"agent_expectation",
        )
        self.experiment_name = experiment_name

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

        if self.continuous_actions:
            model = SAC("CnnPolicy", env.env, verbose=0, **self.policy_config)
        else:
            model = DQN("CnnPolicy", env.env, verbose=0, **self.policy_config)

        callback = CallbackList(
            [
                TimedEvalCallback(
                    env.env,
                    best_model_save_path=self.model_dir,
                    log_path=self.log_dir,
                    eval_freq=max(10, int(self.training_timesteps / 100)),
                    render=False,
                    n_eval_episodes=5,
                ),
                WandbCallback(model_save_path=self.model_dir, verbose=1),
            ]
        )

        model.learn(
            total_timesteps=self.training_timesteps, callback=callback, progress_bar=True
        )

        env.reset_reward_function()

        if self.continuous_actions:
        # actually return best trained model and not last
            return ModelPolicy(SAC.load(self.model_dir/"best_model"))
        else: 
            return ModelPolicy(DQN.load(self.model_dir/"best_model"))


class MDPSolverApproximationVariance(MDPSolverApproximation):
    """
    MDP solver that uses feature expectation and variance matching using an approximation for Soft Value Iteration by using a Soft Critic Actor algorithm

    Parameters
    ----------
    experiment_name : str
        name for the experiment that will be used in naming logging/storage directories
    continuous_actions : bool
        whether or not variance term should be computed (for efficiency reasons will only be computed if necessary)
    policy_config : dict[str, any]
        dictionary containing parameters for SAC/DQN, e.g.,
        buffer_size : int
            buffer size for embedded SAC/DQN training in the approximated SVI
        tau : float
            soft update coefficient ("Polyak update", between 0 and 1)
        train_freq: int
            number of training steps after which the target network should be updated
    training_timesteps : int
            timesteps for embedded SAC/DQN training in the approximated SVI
    T : int
        finite horizon value
    compute_variance : bool
            whether or not variance term should be computed (for efficiency reasons will only be computed if necessary)
    """

    def __init__(
        self,
        experiment_name: str,
        continuous_actions: bool,
        policy_config: dict[str,any],
        training_timesteps: int = 10000,
        T: int = 45,
        compute_variance: bool = True,

    ):
        super().__init__(
            T,
            compute_variance,
            continuous_actions,
            policy_config,
            training_timesteps,
            log_dir=Path("experiments")/experiment_name/"agent_variance",
            model_dir=Path("models")/experiment_name/"agent_variance",
        )
        self.experiment_name = experiment_name

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

        if self.continuous_actions:
            model = SAC("CnnPolicy", env.env, verbose=0, **self.policy_config)
        else:
            model = DQN("CnnPolicy", env.env, verbose=0, **self.policy_config)

        callback = CallbackList(
            [
                TimedEvalCallback(
                    env.env,
                    best_model_save_path=self.model_dir,
                    log_path=self.log_dir,
                    eval_freq=max(10, int(self.training_timesteps / 100)),
                    render=False,
                    n_eval_episodes=5,
                ),
                WandbCallback(model_save_path=self.model_dir, verbose=1),
            ]
        )

        model.learn(
            total_timesteps=self.training_timesteps, callback=callback, progress_bar=True
        )

        env.reset_reward_function()

        if self.continuous_actions:
        # actually return best trained model and not last
            return ModelPolicy(SAC.load(self.model_dir/"best_model"))
        else: 
            return ModelPolicy(DQN.load(self.model_dir/"best_model"))
