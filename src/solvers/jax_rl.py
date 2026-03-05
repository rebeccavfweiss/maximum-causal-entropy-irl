"""
JAX-based RL implementations: DQN (discrete) and SAC (continuous).

Both trainers support warm-starting from previous parameters,
enabling efficient fine-tuning across IRL outer-loop iterations.

Supports both flat (MLP) and image (CNN) observations.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from dataclasses import dataclass, field
from typing import Any
import gymnasium as gym


# ============================================================================
# Shared Components
# ============================================================================


class ReplayBuffer:
    """Simple numpy-backed circular replay buffer supporting flat and image obs."""

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple,
        action_dim: int = 1,
        continuous: bool = False,
    ):
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        self.continuous = continuous

        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        if continuous:
            self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        else:
            self.actions = np.zeros((capacity,), dtype=np.int32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng_key):
        indices = jax.random.randint(rng_key, (batch_size,), 0, self.size)
        indices = np.array(indices)
        return {
            "obs": jnp.array(self.obs[indices]),
            "actions": jnp.array(self.actions[indices]),
            "rewards": jnp.array(self.rewards[indices]),
            "next_obs": jnp.array(self.next_obs[indices]),
            "dones": jnp.array(self.dones[indices]),
        }


@dataclass
class TrainResult:
    """Result from a training run, used for warm-starting."""
    best_params: Any = None
    best_eval_reward: float = -float("inf")
    train_state: dict = field(default_factory=dict)


class RunningMeanStd:
    """Welford's online algorithm for observation normalization."""

    def __init__(self, shape, epsilon=1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0) if x.ndim > 1 else x
        batch_var = np.var(x, axis=0) if x.ndim > 1 else np.zeros_like(x)
        batch_count = x.shape[0] if x.ndim > 1 else 1
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean.astype(np.float32)) / np.sqrt(self.var.astype(np.float32) + 1e-8)


# ============================================================================
# CNN Feature Extractor (for image observations like MiniGrid)
# ============================================================================


class CNNFeatureExtractor(nn.Module):
    """CNN backbone for image observations. Mirrors the PyTorch DynamicMiniGridExtractor."""
    features_dim: int = 128

    @nn.compact
    def __call__(self, x):
        # x shape: (batch, H, W, C) — Flax uses channels-last by default
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # flatten spatial dims
        x = nn.Dense(self.features_dim)(x)
        x = nn.relu(x)
        return x


class CNNQNetwork(nn.Module):
    """Q-network with CNN feature extractor for image observations."""
    action_dim: int
    features_dim: int = 128
    hidden_dims: tuple[int, ...] = (64, 64)

    @nn.compact
    def __call__(self, x):
        x = CNNFeatureExtractor(features_dim=self.features_dim)(x)
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        return nn.Dense(self.action_dim)(x)


class CNNGaussianActor(nn.Module):
    """Gaussian policy with CNN feature extractor for image observations."""
    action_dim: int
    features_dim: int = 128
    hidden_dims: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, x):
        x = CNNFeatureExtractor(features_dim=self.features_dim)(x)
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, -20.0, 2.0)
        return mean, log_std


class CNNTwinQNetwork(nn.Module):
    """Twin Q-networks with CNN feature extractor for image observations (SAC)."""
    features_dim: int = 128
    hidden_dims: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, obs, action):
        features = CNNFeatureExtractor(features_dim=self.features_dim)(obs)
        x = jnp.concatenate([features, action], axis=-1)

        q1 = x
        for dim in self.hidden_dims:
            q1 = nn.Dense(dim)(q1)
            q1 = nn.relu(q1)
        q1 = nn.Dense(1)(q1)

        q2 = x
        for dim in self.hidden_dims:
            q2 = nn.Dense(dim)(q2)
            q2 = nn.relu(q2)
        q2 = nn.Dense(1)(q2)

        return q1.squeeze(-1), q2.squeeze(-1)


# ============================================================================
# DQN
# ============================================================================


class QNetwork(nn.Module):
    """Q-network for discrete action spaces."""
    action_dim: int
    hidden_dims: tuple[int, ...] = (64, 64)

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        return nn.Dense(self.action_dim)(x)


class JaxDQNTrainer:
    """
    DQN trainer with JIT-compiled updates and warm-start support.

    Parameters
    ----------
    obs_dim : int
        Observation space dimension (flat). Ignored if obs_shape is provided.
    action_dim : int
        Number of discrete actions
    config : dict
        Training hyperparameters
    obs_shape : tuple or None
        Full observation shape. If len > 1, uses CNN network.
    """

    def __init__(self, obs_dim: int, action_dim: int, config: dict = None, obs_shape: tuple = None):
        config = config or {}
        self.action_dim = action_dim
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.lr = config.get("lr", 1e-3)
        self.buffer_size = config.get("buffer_size", 50000)
        self.batch_size = config.get("batch_size", 64)
        self.learning_starts = config.get("learning_starts", 1000)
        self.train_freq = config.get("train_freq", 4)
        self.hidden_dims = tuple(config.get("hidden_dims", (64, 64)))
        self.eval_freq = config.get("eval_freq", 5000)
        self.n_eval_episodes = config.get("n_eval_episodes", 5)
        self.features_dim = config.get("features_dim", 128)

        # Epsilon schedule params
        self.epsilon_start = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.05)
        self.epsilon_decay_fraction = config.get("epsilon_decay_fraction", 0.5)
        self.warmstart_epsilon = config.get("warmstart_epsilon", 0.1)

        # Determine observation shape and network type
        if obs_shape is not None and len(obs_shape) > 1:
            self.obs_shape = obs_shape
            self.use_cnn = True
            self.obs_dim = int(np.prod(obs_shape))
            self.network = CNNQNetwork(
                action_dim=action_dim,
                features_dim=self.features_dim,
                hidden_dims=self.hidden_dims,
            )
        else:
            self.obs_shape = (obs_dim,) if obs_shape is None else obs_shape
            self.use_cnn = False
            self.obs_dim = obs_dim
            self.network = QNetwork(action_dim=action_dim, hidden_dims=self.hidden_dims)

    def _get_epsilon(self, step: int, total_steps: int, is_warmstart: bool) -> float:
        if is_warmstart:
            return self.warmstart_epsilon
        decay_steps = int(total_steps * self.epsilon_decay_fraction)
        if step >= decay_steps:
            return self.epsilon_end
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * step / decay_steps

    def _prep_obs(self, obs):
        """Prepare observation for network input (add batch dim if needed)."""
        obs_jax = jnp.array(obs, dtype=jnp.float32)
        if not self.use_cnn:
            obs_jax = obs_jax.flatten()
        return obs_jax

    def _evaluate(self, env, params, obs_normalizer, n_episodes: int) -> float:
        """Evaluate current policy for n_episodes."""
        rewards = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                if obs_normalizer is not None and not self.use_cnn:
                    obs_input = obs_normalizer.normalize(obs)
                else:
                    obs_input = obs
                obs_jax = self._prep_obs(obs_input)
                q_values = self.network.apply(params, obs_jax[None] if self.use_cnn else obs_jax)
                if self.use_cnn:
                    q_values = q_values[0]
                action = int(jnp.argmax(q_values))
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            rewards.append(total_reward)
        return np.mean(rewards)

    def train(
        self,
        env: gym.Env,
        total_timesteps: int,
        initial_state: dict = None,
        obs_normalizer: RunningMeanStd = None,
        custom_reward_fn=None,
        eval_env: gym.Env = None,
        seed: int = 0,
    ) -> TrainResult:
        """
        Main DQN training loop.

        Parameters
        ----------
        env : gym.Env
            Training environment
        total_timesteps : int
            Number of environment steps
        initial_state : dict or None
            If provided, warm-start from these parameters
        obs_normalizer : RunningMeanStd or None
            Observation normalizer (skipped for CNN)
        custom_reward_fn : callable or None
            Custom reward function r(obs) overriding env rewards
        eval_env : gym.Env or None
            Separate env for evaluation (uses env if None)
        seed : int
            Random seed

        Returns
        -------
        TrainResult with best_params, best_eval_reward, and train_state
        """
        rng = jax.random.PRNGKey(seed)
        is_warmstart = initial_state is not None

        # Initialize or restore network params
        if initial_state is not None:
            params = initial_state["params"]
            target_params = initial_state["target_params"]
            opt_state = initial_state["opt_state"]
            optimizer = optax.adam(self.lr)
            opt_state = optimizer.init(params)
        else:
            rng, init_rng = jax.random.split(rng)
            if self.use_cnn:
                dummy_obs = jnp.zeros((1, *self.obs_shape))
            else:
                dummy_obs = jnp.zeros((1, self.obs_dim))
            params = self.network.init(init_rng, dummy_obs)
            target_params = jax.tree.map(lambda p: p.copy(), params)
            optimizer = optax.adam(self.lr)
            opt_state = optimizer.init(params)

        buffer = ReplayBuffer(self.buffer_size, self.obs_shape)
        eval_env = eval_env or env

        best_params = jax.tree.map(lambda p: p.copy(), params)
        best_eval_reward = -float("inf")

        # Define loss and update as closures for JIT
        @jax.jit
        def train_step(params, target_params, opt_state, batch, rng_key):
            def loss_fn(p):
                q_values = self.network.apply(p, batch["obs"])
                q_selected = q_values[jnp.arange(q_values.shape[0]), batch["actions"]]

                next_q = self.network.apply(target_params, batch["next_obs"])
                next_q_max = jnp.max(next_q, axis=-1)
                targets = batch["rewards"] + self.gamma * (1.0 - batch["dones"]) * next_q_max

                return jnp.mean((q_selected - jax.lax.stop_gradient(targets)) ** 2)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss

        @jax.jit
        def polyak_update(params, target_params):
            return jax.tree.map(
                lambda p, tp: self.tau * p + (1 - self.tau) * tp,
                params, target_params,
            )

        # Training loop
        obs, _ = env.reset()
        for step in range(total_timesteps):
            epsilon = self._get_epsilon(step, total_timesteps, is_warmstart)
            rng, action_rng, sample_rng = jax.random.split(rng, 3)

            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                if obs_normalizer is not None and not self.use_cnn:
                    obs_input = obs_normalizer.normalize(obs)
                else:
                    obs_input = obs
                obs_jax = self._prep_obs(obs_input)
                q_values = self.network.apply(params, obs_jax[None] if self.use_cnn else obs_jax)
                if self.use_cnn:
                    q_values = q_values[0]
                action = int(jnp.argmax(q_values))

            next_obs, reward, terminated, truncated, info = env.step(action)

            if custom_reward_fn is not None:
                reward = custom_reward_fn(next_obs)

            # Update normalizer (only for flat obs)
            if obs_normalizer is not None and not self.use_cnn:
                obs_normalizer.update(obs.reshape(1, -1))

            # Normalize for buffer storage (skip for CNN — use raw pixel values)
            if obs_normalizer is not None and not self.use_cnn:
                obs_store = obs_normalizer.normalize(obs)
                next_obs_store = obs_normalizer.normalize(next_obs)
            else:
                obs_store = obs
                next_obs_store = next_obs

            buffer.add(obs_store, action, reward, next_obs_store, terminated or truncated)

            if terminated or truncated:
                obs, _ = env.reset()
            else:
                obs = next_obs

            # Train
            if step >= self.learning_starts and step % self.train_freq == 0:
                batch = buffer.sample(self.batch_size, sample_rng)
                params, opt_state, loss = train_step(params, target_params, opt_state, batch, sample_rng)
                target_params = polyak_update(params, target_params)

            # Evaluate
            if step > 0 and step % self.eval_freq == 0:
                eval_reward = self._evaluate(eval_env, params, obs_normalizer, self.n_eval_episodes)
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    best_params = jax.tree.map(lambda p: p.copy(), params)

        # Final evaluation
        eval_reward = self._evaluate(eval_env, params, obs_normalizer, self.n_eval_episodes)
        if eval_reward > best_eval_reward:
            best_eval_reward = eval_reward
            best_params = jax.tree.map(lambda p: p.copy(), params)

        return TrainResult(
            best_params=best_params,
            best_eval_reward=best_eval_reward,
            train_state={
                "params": params,
                "target_params": target_params,
                "opt_state": opt_state,
            },
        )


# ============================================================================
# SAC
# ============================================================================


class GaussianActor(nn.Module):
    """Gaussian policy for continuous action spaces (squashed via tanh)."""
    action_dim: int
    hidden_dims: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, -20.0, 2.0)
        return mean, log_std


class TwinQNetwork(nn.Module):
    """Twin Q-networks for SAC (clipped double-Q)."""
    hidden_dims: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        # Q1
        q1 = x
        for dim in self.hidden_dims:
            q1 = nn.Dense(dim)(q1)
            q1 = nn.relu(q1)
        q1 = nn.Dense(1)(q1)

        # Q2
        q2 = x
        for dim in self.hidden_dims:
            q2 = nn.Dense(dim)(q2)
            q2 = nn.relu(q2)
        q2 = nn.Dense(1)(q2)

        return q1.squeeze(-1), q2.squeeze(-1)


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def _sample_action(actor, actor_params, obs, rng_key):
    """Sample action from squashed Gaussian policy, return (action, log_prob)."""
    mean, log_std = actor.apply(actor_params, obs)
    std = jnp.exp(log_std)
    noise = jax.random.normal(rng_key, mean.shape)
    x_t = mean + std * noise  # pre-squash
    action = jnp.tanh(x_t)

    # Log probability with tanh correction
    log_prob = -0.5 * (((x_t - mean) / (std + 1e-8)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))
    log_prob = jnp.sum(log_prob, axis=-1)
    # Tanh squashing correction
    log_prob -= jnp.sum(jnp.log(1 - action**2 + 1e-6), axis=-1)

    return action, log_prob


class JaxSACTrainer:
    """
    SAC trainer with JIT-compiled updates and warm-start support.

    Parameters
    ----------
    obs_dim : int
        Observation space dimension (flat). Ignored if obs_shape is provided.
    action_dim : int
        Action space dimension
    config : dict
        Training hyperparameters
    obs_shape : tuple or None
        Full observation shape. If len > 1, uses CNN networks.
    """

    def __init__(self, obs_dim: int, action_dim: int, config: dict = None, obs_shape: tuple = None):
        config = config or {}
        self.action_dim = action_dim
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.actor_lr = config.get("actor_lr", 3e-4)
        self.critic_lr = config.get("critic_lr", 3e-4)
        self.alpha_lr = config.get("alpha_lr", 3e-4)
        self.buffer_size = config.get("buffer_size", 100000)
        self.batch_size = config.get("batch_size", 256)
        self.learning_starts = config.get("learning_starts", 1000)
        self.train_freq = config.get("train_freq", 1)
        self.hidden_dims = tuple(config.get("hidden_dims", (256, 256)))
        self.eval_freq = config.get("eval_freq", 5000)
        self.n_eval_episodes = config.get("n_eval_episodes", 5)
        self.init_alpha = config.get("init_alpha", 1.0)
        self.features_dim = config.get("features_dim", 128)

        self.target_entropy = -action_dim

        # Determine observation shape and network type
        if obs_shape is not None and len(obs_shape) > 1:
            self.obs_shape = obs_shape
            self.use_cnn = True
            self.obs_dim = int(np.prod(obs_shape))
            self.actor = CNNGaussianActor(
                action_dim=action_dim,
                features_dim=self.features_dim,
                hidden_dims=self.hidden_dims,
            )
            self.critic = CNNTwinQNetwork(
                features_dim=self.features_dim,
                hidden_dims=self.hidden_dims,
            )
        else:
            self.obs_shape = (obs_dim,) if obs_shape is None else obs_shape
            self.use_cnn = False
            self.obs_dim = obs_dim
            self.actor = GaussianActor(action_dim=action_dim, hidden_dims=self.hidden_dims)
            self.critic = TwinQNetwork(hidden_dims=self.hidden_dims)

    def _prep_obs(self, obs):
        """Prepare observation for network input."""
        obs_jax = jnp.array(obs, dtype=jnp.float32)
        if not self.use_cnn:
            obs_jax = obs_jax.flatten()
        return obs_jax

    def _evaluate(self, env, actor_params, obs_normalizer, n_episodes: int) -> float:
        """Evaluate current policy deterministically."""
        rewards = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                if obs_normalizer is not None and not self.use_cnn:
                    obs_input = obs_normalizer.normalize(obs)
                else:
                    obs_input = obs
                obs_jax = self._prep_obs(obs_input)
                mean, _ = self.actor.apply(actor_params, obs_jax[None] if self.use_cnn else obs_jax)
                if self.use_cnn:
                    mean = mean[0]
                action = np.array(jnp.tanh(mean))
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            rewards.append(total_reward)
        return np.mean(rewards)

    def train(
        self,
        env: gym.Env,
        total_timesteps: int,
        initial_state: dict = None,
        obs_normalizer: RunningMeanStd = None,
        custom_reward_fn=None,
        eval_env: gym.Env = None,
        seed: int = 0,
    ) -> TrainResult:
        """
        Main SAC training loop.

        Parameters
        ----------
        env : gym.Env
            Training environment
        total_timesteps : int
            Number of environment steps
        initial_state : dict or None
            If provided, warm-start from these params
        obs_normalizer : RunningMeanStd or None
            Observation normalizer (skipped for CNN)
        custom_reward_fn : callable or None
            Custom reward function r(obs) overriding env rewards
        eval_env : gym.Env or None
            Separate env for evaluation
        seed : int
            Random seed

        Returns
        -------
        TrainResult with best_params and train_state
        """
        rng = jax.random.PRNGKey(seed)
        eval_env = eval_env or env

        # Optimizers
        actor_optimizer = optax.adam(self.actor_lr)
        critic_optimizer = optax.adam(self.critic_lr)
        alpha_optimizer = optax.adam(self.alpha_lr)

        # Initialize or restore
        if initial_state is not None:
            actor_params = initial_state["actor_params"]
            critic_params = initial_state["critic_params"]
            target_critic_params = initial_state["target_critic_params"]
            log_alpha = initial_state["log_alpha"]
            actor_opt_state = actor_optimizer.init(actor_params)
            critic_opt_state = critic_optimizer.init(critic_params)
            alpha_opt_state = alpha_optimizer.init(log_alpha)
        else:
            rng, actor_rng, critic_rng = jax.random.split(rng, 3)
            if self.use_cnn:
                dummy_obs = jnp.zeros((1, *self.obs_shape))
            else:
                dummy_obs = jnp.zeros((1, self.obs_dim))
            dummy_action = jnp.zeros((1, self.action_dim))

            actor_params = self.actor.init(actor_rng, dummy_obs)
            critic_params = self.critic.init(critic_rng, dummy_obs, dummy_action)
            target_critic_params = jax.tree.map(lambda p: p.copy(), critic_params)
            log_alpha = jnp.array(jnp.log(self.init_alpha), dtype=jnp.float32)

            actor_opt_state = actor_optimizer.init(actor_params)
            critic_opt_state = critic_optimizer.init(critic_params)
            alpha_opt_state = alpha_optimizer.init(log_alpha)

        buffer = ReplayBuffer(
            self.buffer_size, self.obs_shape,
            action_dim=self.action_dim, continuous=True,
        )

        best_actor_params = jax.tree.map(lambda p: p.copy(), actor_params)
        best_eval_reward = -float("inf")

        # JIT-compiled update functions
        actor_net = self.actor
        critic_net = self.critic
        gamma = self.gamma
        target_entropy = self.target_entropy
        tau = self.tau

        @jax.jit
        def update_critic(critic_params, target_critic_params, actor_params,
                          log_alpha, critic_opt_state, batch, rng_key):
            alpha = jnp.exp(log_alpha)

            def critic_loss_fn(cp):
                next_actions, next_log_probs = _sample_action(
                    actor_net, actor_params, batch["next_obs"], rng_key,
                )
                tq1, tq2 = critic_net.apply(target_critic_params, batch["next_obs"], next_actions)
                target_q = jnp.minimum(tq1, tq2) - alpha * next_log_probs
                targets = batch["rewards"] + gamma * (1.0 - batch["dones"]) * target_q

                q1, q2 = critic_net.apply(cp, batch["obs"], batch["actions"])
                loss = jnp.mean((q1 - jax.lax.stop_gradient(targets)) ** 2) + \
                       jnp.mean((q2 - jax.lax.stop_gradient(targets)) ** 2)
                return loss

            loss, grads = jax.value_and_grad(critic_loss_fn)(critic_params)
            updates, new_opt = critic_optimizer.update(grads, critic_opt_state, critic_params)
            new_critic = optax.apply_updates(critic_params, updates)
            return new_critic, new_opt, loss

        @jax.jit
        def update_actor(actor_params, critic_params, log_alpha,
                         actor_opt_state, batch, rng_key):
            alpha = jnp.exp(log_alpha)

            def actor_loss_fn(ap):
                actions, log_probs = _sample_action(actor_net, ap, batch["obs"], rng_key)
                q1, q2 = critic_net.apply(critic_params, batch["obs"], actions)
                min_q = jnp.minimum(q1, q2)
                return jnp.mean(alpha * log_probs - min_q)

            loss, grads = jax.value_and_grad(actor_loss_fn)(actor_params)
            updates, new_opt = actor_optimizer.update(grads, actor_opt_state, actor_params)
            new_actor = optax.apply_updates(actor_params, updates)
            return new_actor, new_opt, loss

        @jax.jit
        def update_alpha(log_alpha, actor_params, alpha_opt_state, batch, rng_key):
            actions, log_probs = _sample_action(actor_net, actor_params, batch["obs"], rng_key)

            def alpha_loss_fn(la):
                alpha = jnp.exp(la)
                return jnp.mean(-alpha * jax.lax.stop_gradient(log_probs + target_entropy))

            loss, grad = jax.value_and_grad(alpha_loss_fn)(log_alpha)
            updates, new_opt = alpha_optimizer.update(grad, alpha_opt_state)
            new_log_alpha = optax.apply_updates(log_alpha, updates)
            return new_log_alpha, new_opt, loss

        @jax.jit
        def polyak_update(params, target_params):
            return jax.tree.map(
                lambda p, tp: tau * p + (1 - tau) * tp,
                params, target_params,
            )

        # Training loop
        obs, _ = env.reset()
        for step in range(total_timesteps):
            rng, action_rng, sample_rng, critic_rng, actor_rng, alpha_rng = jax.random.split(rng, 6)

            if step < self.learning_starts and initial_state is None:
                action = env.action_space.sample()
            else:
                if obs_normalizer is not None and not self.use_cnn:
                    obs_input = obs_normalizer.normalize(obs)
                else:
                    obs_input = obs
                obs_jax = self._prep_obs(obs_input)
                action, _ = _sample_action(
                    actor_net, actor_params,
                    obs_jax[None] if self.use_cnn else obs_jax,
                    action_rng,
                )
                if self.use_cnn:
                    action = action[0]
                action = np.array(action)
                action = np.clip(action, env.action_space.low, env.action_space.high)

            next_obs, reward, terminated, truncated, info = env.step(action)

            if custom_reward_fn is not None:
                reward = custom_reward_fn(next_obs)

            if obs_normalizer is not None and not self.use_cnn:
                obs_normalizer.update(obs.reshape(1, -1))

            if obs_normalizer is not None and not self.use_cnn:
                obs_store = obs_normalizer.normalize(obs)
                next_obs_store = obs_normalizer.normalize(next_obs)
            else:
                obs_store = obs
                next_obs_store = next_obs

            buffer.add(obs_store, action, reward, next_obs_store, terminated or truncated)

            if terminated or truncated:
                obs, _ = env.reset()
            else:
                obs = next_obs

            # Train
            if step >= self.learning_starts and step % self.train_freq == 0:
                batch = buffer.sample(self.batch_size, sample_rng)

                critic_params, critic_opt_state, _ = update_critic(
                    critic_params, target_critic_params, actor_params,
                    log_alpha, critic_opt_state, batch, critic_rng,
                )

                actor_params, actor_opt_state, _ = update_actor(
                    actor_params, critic_params, log_alpha,
                    actor_opt_state, batch, actor_rng,
                )

                log_alpha, alpha_opt_state, _ = update_alpha(
                    log_alpha, actor_params, alpha_opt_state, batch, alpha_rng,
                )

                target_critic_params = polyak_update(critic_params, target_critic_params)

            # Evaluate
            if step > 0 and step % self.eval_freq == 0:
                eval_reward = self._evaluate(eval_env, actor_params, obs_normalizer, self.n_eval_episodes)
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    best_actor_params = jax.tree.map(lambda p: p.copy(), actor_params)

        # Final evaluation
        eval_reward = self._evaluate(eval_env, actor_params, obs_normalizer, self.n_eval_episodes)
        if eval_reward > best_eval_reward:
            best_eval_reward = eval_reward
            best_actor_params = jax.tree.map(lambda p: p.copy(), actor_params)

        return TrainResult(
            best_params=best_actor_params,
            best_eval_reward=best_eval_reward,
            train_state={
                "actor_params": actor_params,
                "critic_params": critic_params,
                "target_critic_params": target_critic_params,
                "log_alpha": log_alpha,
                "actor_opt_state": actor_opt_state,
                "critic_opt_state": critic_opt_state,
                "alpha_opt_state": alpha_opt_state,
            },
        )
