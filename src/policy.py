from abc import ABC, abstractmethod
import numpy as np
from stable_baselines3 import PPO
import jax.numpy as jnp


class Policy(ABC):
    """
    Wrapper class for differnt kinds of policies to offer the same interface to other parts of the algorithm no matter what the policy looks like.

    """

    @abstractmethod
    def predict(obs, t: int = None) -> int:
        pass


class TabularPolicy(Policy):
    """
    Specific implementation of the Policy interface to use a tablular form of a policy

    Parameters
    ----------
    pi : nd.array
        tabluar policy to use
    """

    def __init__(self, pi: np.ndarray):
        self.pi = pi

    def predict(self, obs, t: int = None) -> int:
        """
        Predicts an action given an observation in the environment

        Parameters
        ----------
        obs
            observation from the environment
        t : int
            time step (only relevant if policy time dependent)

        Returns
        -------
        action : int
            action to take in the given state
        """

        probs = self.pi[t, obs]
        if probs.sum() != 1.0:
            probs /= probs.sum()
        action = int(np.random.choice(np.arange(len(probs)), p=probs))

        return action


class ModelPolicy(Policy):
    """
    Specific implementation of the Policy interface to use a model for the policy, e.g., Neural networks trained with PPO

    Parameters
    ----------
    model : PPO
        model to use for prediction
    """

    def __init__(self, model: PPO):
        self.model = model

    def predict(self, obs, t: int = None) -> int:
        """
        Predicts an action given an observation in the environment

        Parameters
        ----------
        obs
            observation from the environment
        t : int
            time step (only relevant if policy time dependent)

        Returns
        -------
        action : int
            action to take in the given state
        """

        return self.model.predict(obs, deterministic=True)[0]


class JaxPolicy(Policy):
    """
    Wraps JAX network parameters for inference via the Policy interface.
    Works for both DQN (discrete) and SAC (continuous) policies.

    Parameters
    ----------
    params : Any
        JAX/Flax network parameters
    network : nn.Module
        Flax network module (QNetwork or GaussianActor)
    obs_normalizer : RunningMeanStd or None
        Observation normalizer
    mode : str
        "dqn" for discrete actions, "sac" for continuous actions
    """

    def __init__(self, params, network, obs_normalizer=None, mode="dqn", use_cnn=False):
        self.params = params
        self.network = network
        self.obs_normalizer = obs_normalizer
        self.mode = mode
        self.use_cnn = use_cnn

    def predict(self, obs, t: int = None):
        if self.use_cnn:
            # Keep spatial shape, add batch dim: (H, W, C) -> (1, H, W, C)
            obs_jax = jnp.array(obs, dtype=jnp.float32)[None]
        else:
            if isinstance(obs, np.ndarray):
                obs_jax = jnp.array(obs.flatten(), dtype=jnp.float32)
            else:
                obs_jax = jnp.array(obs, dtype=jnp.float32).flatten()

            if self.obs_normalizer is not None:
                obs_jax = jnp.array(self.obs_normalizer.normalize(np.array(obs_jax)))

        if self.mode == "sac":
            mean, _ = self.network.apply(self.params, obs_jax)
            if self.use_cnn:
                mean = mean[0]
            return np.array(jnp.tanh(mean))
        else:
            q_values = self.network.apply(self.params, obs_jax)
            if self.use_cnn:
                q_values = q_values[0]
            return int(jnp.argmax(q_values))
