import agents.learner as learner
import agents.demonstrator as demonstrator
from environments.continuous_minigrid_environment import ContinuousMinigridEnvironment
from solvers.MDP_solver_approximation import (
    MDPSolverApproximationExpectation,
    MDPSolverApproximationVariance,
)
import numpy as np
import psutil
import gc
import wandb
from random import randint
import minigrid
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def create_minigrid_env(
    grid_size: int = 9, T: int = 25, env_name: str = "MiniGrid-DoorKey-5x5-v0"
):

    config_env = {
        "gamma": 1.0,
        "env_name": env_name,
        "render_mode": "rgb_array",
        "grid_size": grid_size,
        "seed": None,
        "T": T,
    }

    env = ContinuousMinigridEnvironment(config_env)
    return env


def create_config_learner(n_trajectories: int = 1, maxiter: int = 3):
    config_default_learner = {
        "tol_exp": 50.0,
        "tol_var": 1250.0,
        "miniter": 1,
        "maxiter": maxiter,
        "n_trajectories": n_trajectories,
    }

    return config_default_learner


def log_memory(stage=""):
    mem = psutil.virtual_memory()
    wandb.log(
        {
            f"memory_free_mb_{stage}": mem.available / (1024**2),
            f"memory_used_percent_{stage}": mem.percent,
            f"memory_total_mb_{stage}": mem.total / (1024**2),
        }
    )


class DynamicMiniGridExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        # We don't know the flatten size yet, so we call super() first
        super().__init__(observation_space, features_dim)

        # MiniGrid observations are typically (channels, height, width)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # --- DYNAMIC CALCULATION ---
        # 1. Create a fake image matching the observation space shape
        # 2. Pass it through the CNN layers defined above
        # 3. Check the output shape
        with th.no_grad():
            sample_tensor = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_tensor).shape[1]

        # Use that calculated n_flatten to build the linear head
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))


if __name__ == "__main__":

    show = False
    store = True
    env_name = "MiniGrid-DoorKey-5x5-v0"
    demo_training_algorithm = "ppo"
    agent_training_algorithm = "sac"

    grid_size = 5
    T = 70

    maxiter = 50
    n_trajectories = 150
    training_timesteps = 350000
    policy_config = dict(
        buffer_size=50000, tau=0.005, gamma=1.0, train_freq=5, device="auto"
    )

    policy_kwargs = dict(
        features_extractor_class=DynamicMiniGridExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    learning_rate = lambda step: max(0.975 ** (step + 1), 0.01)

    wandb.init(
        project=f"mceirl-minigrid-{env_name}",
        name=f"{env_name}-g{grid_size}-T{T}-iter{maxiter}-sac_iter{training_timesteps}-traj{n_trajectories}",
        config={
            "maxiter": maxiter,
            "n_trajectories": n_trajectories,
            "sac_dqn_timesteps": training_timesteps,
            "sac_dqn_buffer_size": policy_config["buffer_size"],
            "T": T,
        },
    )

    log_memory("start")

    # create the environment
    env = create_minigrid_env(grid_size=grid_size, env_name=env_name)

    # Learner config
    config_default_learner = create_config_learner(n_trajectories, maxiter)

    log_memory("env_config_creation")

    # create demonstrator
    demo = demonstrator.ContinuousDemonstrator(
        env,
        demonstrator_name="MinigridDemonstrator",
        T=T,
        n_trajectories=n_trajectories,
        training_algorithm=demo_training_algorithm,
        solver=MDPSolverApproximationExpectation(
            experiment_name=env_name,
            training_algorithm=demo_training_algorithm,
            T=T,
            compute_variance=True,
            policy_config=policy_config,
            policy_kwargs=policy_kwargs,
            training_timesteps=training_timesteps,
        ),
        policy_kwargs=policy_kwargs,
        time_steps=7_500_000,
    )

    log_memory("demonstrator_creation")

    reward_demonstrator = env.compute_true_reward_for_agent(demo, n_trajectories, T)

    wandb.log(
        {
            "demonstrator_expected_value": demo.mu_demonstrator[0],
            "demonstrator_variance": demo.mu_demonstrator[1],
            "demonstrator_reward": reward_demonstrator,
        }
    )

    path_to_file = demo.render(show, store, 0)

    if path_to_file is not None:
        # log a video to see how the demonstrator is performing
        wandb.log(
            {
                f"eval/video_{demo.agent_name}": wandb.Video(
                    path_to_file, fps=4, format="mp4"
                )
            }
        )

    # clean up
    del demo.policy
    log_memory("demonstrator_policy_cleanup")

    # create agent that also matches variances
    agent_variance = learner.ApproximateLearner(
        env,
        demo.mu_demonstrator,
        config_default_learner,
        agent_name="AgentVariance",
        solver=MDPSolverApproximationVariance(
            experiment_name=env_name,
            training_algorithm=agent_training_algorithm,
            T=T,
            compute_variance=True,
            policy_config=policy_config,
            policy_kwargs=policy_kwargs,
            training_timesteps=training_timesteps,
        ),
        learning_rate=learning_rate,
    )

    iter_variance, time_variance = agent_variance.batch_MCE()
    agent_variance.compute_and_draw(show, store, 4)
    reward_variance = env.compute_true_reward_for_agent(
        agent_variance, n_trajectories, T
    )

    log_memory("agent_variance_finished")

    artifact = wandb.Artifact(f"agent_variance_model", type="model")
    artifact.add_file(agent_variance.solver.model_dir / "best_model.zip")
    wandb.log_artifact(artifact)

    # Free up memory
    del agent_variance.policy
    log_memory("agent_expectation_policy_cleanup")

    wandb.log(
        {
            "reward_variance": reward_variance,
            "reward_diff_variance": np.abs(reward_demonstrator - reward_variance),
            "iterations_variance": iter_variance,
            "time_total_variance": sum(time_variance),
            "time_avg_per_iter_variance": np.mean(time_variance),
        }
    )

    # create agent that uses only expectation matching
    agent_expectation = learner.ApproximateLearner(
        env,
        demo.mu_demonstrator,
        config_default_learner,
        agent_name="AgentExpectation",
        solver=MDPSolverApproximationExpectation(
            training_algorithm=agent_training_algorithm,
            experiment_name=env_name,
            T=T,
            compute_variance=False,
            policy_config=policy_config,
            training_timesteps=training_timesteps,
        ),
        learning_rate=learning_rate,
    )

    iter_expectation, time_expectation = agent_expectation.batch_MCE()
    agent_expectation.compute_and_draw(show, store, 2)
    reward_expectation = env.compute_true_reward_for_agent(
        agent_expectation, n_trajectories, T
    )

    log_memory("agent_expectation_finished")

    artifact = wandb.Artifact(f"agent_expectation_model", type="model")
    artifact.add_file(agent_expectation.solver.model_dir / "best_model.zip")
    wandb.log_artifact(artifact)

    wandb.log(
        {
            "reward_expectation": reward_expectation,
            "reward_diff_expectation": np.abs(reward_demonstrator - reward_expectation),
            "iterations_expectation": iter_expectation,
            "time_total_expectation": sum(time_expectation),
            "time_avg_per_iter_expectation": np.mean(time_expectation),
        }
    )

    del agent_expectation
    gc.collect()
    log_memory("agent_expectation_policy_cleanup")

    wandb.finish()
