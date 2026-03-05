"""
JAX-based MDP solvers that parallel MDPSolverApproximationExpectation/Variance
but use JaxDQNTrainer or JaxSACTrainer instead of stable-baselines3.

Key difference: soft_value_iteration accepts and returns network parameters
for warm-starting across IRL outer-loop iterations.
"""

import numpy as np
from pathlib import Path
from solvers.MDP_solver import MDPSolver
from solvers.MDP_solver_approximation import MDPSolverApproximation
from solvers.jax_rl import JaxDQNTrainer, JaxSACTrainer, RunningMeanStd
from environments.environment import Environment
from policy import Policy, JaxPolicy
import gymnasium as gym


class JaxSolver(MDPSolverApproximation):
    """
    Base JAX solver. Replaces MDPSolverApproximation for JAX-based training.

    Parameters
    ----------
    T : int
        Finite horizon value
    compute_variance : bool
        Whether variance term should be computed
    training_algorithm : str
        "dqn" for discrete actions, "sac" for continuous actions
    training_config : dict
        Hyperparameters for the DQN/SAC trainer
    experiment_name : str
        Name for logging directories
    full_training_timesteps : int
        Timesteps for training from scratch (first iteration)
    finetune_timesteps : int
        Timesteps for warm-start fine-tuning (subsequent iterations)
    """

    def __init__(
        self,
        T: int,
        compute_variance: bool,
        training_algorithm: str,
        training_config: dict = None,
        experiment_name: str = "jax_experiment",
        full_training_timesteps: int = 200_000,
        finetune_timesteps: int = 5_000,
        log_dir: str = None,
        model_dir: str = None,
    ):
        # Initialize parent without SB3-specific policy_config
        super().__init__(
            T=T,
            compute_variance=compute_variance,
            training_algorithm=training_algorithm,
            policy_config={},
            policy_kwargs=None,
            training_timesteps=full_training_timesteps,
            log_dir=log_dir,
            model_dir=model_dir,
        )

        self.training_config = training_config or {}
        self.experiment_name = experiment_name
        self.full_training_timesteps = full_training_timesteps
        self.finetune_timesteps = finetune_timesteps
        self.trainer = None
        self.obs_normalizer = None

    def _create_trainer(self, env: gym.Env):
        """Lazily create the appropriate trainer based on training_algorithm."""
        obs_shape = env.observation_space.shape
        obs_dim = obs_shape[0] if len(obs_shape) == 1 else int(np.prod(obs_shape))
        use_cnn = len(obs_shape) > 1

        if self.training_algorithm == "sac":
            action_dim = env.action_space.shape[0]
            self.trainer = JaxSACTrainer(
                obs_dim, action_dim, self.training_config, obs_shape=obs_shape
            )
        else:
            action_dim = env.action_space.n
            self.trainer = JaxDQNTrainer(
                obs_dim, action_dim, self.training_config, obs_shape=obs_shape
            )

        if use_cnn:
            self.obs_normalizer = None
        else:
            self.obs_normalizer = RunningMeanStd(shape=(obs_dim,))


class JaxSolverExpectation(JaxSolver):
    """
    JAX solver for expectation-only matching.
    Reward: r(s) = theta_e * phi(s)
    """

    def __init__(
        self,
        experiment_name: str,
        training_algorithm: str = "dqn",
        training_config: dict = None,
        T: int = 45,
        compute_variance: bool = False,
        full_training_timesteps: int = 200_000,
        finetune_timesteps: int = 5_000,
    ):
        super().__init__(
            T=T,
            compute_variance=compute_variance,
            training_algorithm=training_algorithm,
            training_config=training_config,
            experiment_name=experiment_name,
            full_training_timesteps=full_training_timesteps,
            finetune_timesteps=finetune_timesteps,
            log_dir=str(
                Path("experiments") / experiment_name / "jax_agent_expectation"
            ),
            model_dir=str(Path("models") / experiment_name / "jax_agent_expectation"),
        )

    def soft_value_iteration(
        self,
        env: Environment,
        values: dict,
        prev_params: dict = None,
    ) -> tuple[Policy, dict]:
        """
        Train policy via DQN or SAC to maximize the custom reward function.

        Parameters
        ----------
        env : Environment
            The environment
        values : dict
            dict with "reward" function
        prev_params : dict or None
            If provided, warm-start from these parameters

        Returns
        -------
        policy : JaxPolicy
        train_state : dict for warm-starting next iteration
        """
        # Set custom reward: r(s) = reward(s)
        env.set_custom_reward_function(lambda s: values["reward"](s.flatten()))

        gym_env = env.get_gym_env()

        if self.trainer is None:
            self._create_trainer(gym_env)

        # Determine training steps
        timesteps = (
            self.finetune_timesteps
            if prev_params is not None
            else self.full_training_timesteps
        )

        result = self.trainer.train(
            env=gym_env,
            total_timesteps=timesteps,
            initial_state=prev_params,
            obs_normalizer=self.obs_normalizer,
            custom_reward_fn=lambda s: values["reward"](s.flatten()),
        )

        env.reset_reward_function()

        # Create appropriate JaxPolicy
        mode = "sac" if self.training_algorithm == "sac" else "dqn"
        network = self.trainer.actor if mode == "sac" else self.trainer.network
        policy = JaxPolicy(
            result.best_params,
            network,
            obs_normalizer=self.obs_normalizer,
            mode=mode,
            use_cnn=self.trainer.use_cnn,
        )

        return policy, result.train_state


class JaxSolverVariance(JaxSolver):
    """
    JAX solver for expectation + variance matching.
    Reward: r(s) = theta_e * phi(s) + phi(s)^T theta_v phi(s)
    """

    def __init__(
        self,
        experiment_name: str,
        training_algorithm: str = "dqn",
        training_config: dict = None,
        T: int = 45,
        compute_variance: bool = True,
        full_training_timesteps: int = 200_000,
        finetune_timesteps: int = 5_000,
    ):
        super().__init__(
            T=T,
            compute_variance=compute_variance,
            training_algorithm=training_algorithm,
            training_config=training_config,
            experiment_name=experiment_name,
            full_training_timesteps=full_training_timesteps,
            finetune_timesteps=finetune_timesteps,
            log_dir=str(Path("experiments") / experiment_name / "jax_agent_variance"),
            model_dir=str(Path("models") / experiment_name / "jax_agent_variance"),
        )

    def soft_value_iteration(
        self,
        env: Environment,
        values: dict,
        prev_params: dict = None,
    ) -> tuple[Policy, dict]:
        """
        Train policy via DQN or SAC to maximize reward + variance.

        Parameters
        ----------
        env : Environment
        values : dict with "reward" and "variance" functions
        prev_params : dict or None

        Returns
        -------
        policy : JaxPolicy
        train_state : dict for warm-starting
        """

        # Combined reward: r(s) = reward(s) + variance(s)
        def combined_reward(s):
            s_flat = s.flatten()
            return values["reward"](s_flat) + values["variance"](s_flat)

        env.set_custom_reward_function(combined_reward)

        gym_env = env.get_gym_env()

        if self.trainer is None:
            self._create_trainer(gym_env)

        timesteps = (
            self.finetune_timesteps
            if prev_params is not None
            else self.full_training_timesteps
        )

        result = self.trainer.train(
            env=gym_env,
            total_timesteps=timesteps,
            initial_state=prev_params,
            obs_normalizer=self.obs_normalizer,
            custom_reward_fn=combined_reward,
        )

        env.reset_reward_function()

        mode = "sac" if self.training_algorithm == "sac" else "dqn"
        network = self.trainer.actor if mode == "sac" else self.trainer.network
        policy = JaxPolicy(
            result.best_params,
            network,
            obs_normalizer=self.obs_normalizer,
            mode=mode,
            use_cnn=self.trainer.use_cnn,
        )

        return policy, result.train_state
