import agents.learner as learner
import agents.demonstrator as demonstrator
from environments.box2d_environment import Box2DEnvironment
from solvers.MDP_solver_approximation import (
    MDPSolverApproximationExpectation,
    MDPSolverApproximationVariance,
)
import numpy as np
import psutil
import gc
import wandb
import torch


def create_box2d_env(
    env_id: str = "LunarLander-v3",
    T: int = 1000,
    gamma: float = 0.99,
):

    config_env = {
        "env_id": env_id,
        "gamma": gamma,
        "T": T,
        "continuous": False,
        "enable_wind": False,
    }

    env = Box2DEnvironment(config_env)
    return env


def create_config_learner(n_trajectories: int = 1, maxiter: int = 3):
    config_default_learner = {
        "tol_exp": 5.0,
        "tol_var": 250.0,
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


if __name__ == "__main__":

    show = False
    store = True
    continuous = False
    # env_id = "LunarLander-v3"
    env_id = "BipedalWalker-v3"

    experiment_name = env_id + ("_continuous" if continuous else "_discrete")
    if env_id == "LunarLander-v3":
        repo_id = "Chiz"
    else:
        repo_id = "matamaki"

    demo_training_algorithm = "ppo"
    agent_training_algorithm = "sac" if continuous else "dqn"

    maxiter = 50
    n_trajectories = 150
    training_timesteps = 350000
    demo_policy_config = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[256, 256],
        gamma=1.0,
    )
    agent_policy_config = dict(
        policy="MlpPolicy",
        buffer_size=50000,
        tau=0.005,
        gamma=1.0,
        train_freq=5,
        device="auto",
    )

    T = 800 if env_id == "LunarLander-v3" else 300

    learning_rate = lambda step: max(0.975 ** (step + 1), 0.01)

    wandb.init(
        project=f"mceirl-{env_id}",
        name=f"{experiment_name}-iter{maxiter}-sac_iter{training_timesteps}-T{T}-traj{n_trajectories}",
        config={
            "maxiter": maxiter,
            "n_trajectories": n_trajectories,
            "sac_dqn_timesteps": training_timesteps,
            "T": T,
            "actions_space": (
                "continous action space" if continuous else "discrete action space"
            ),
        },
    )

    log_memory("start")

    # create the environment
    env = create_box2d_env(
        env_id=env_id,
        T=T,
        gamma=demo_policy_config["gamma"],
    )

    # Learner config
    config_default_learner = create_config_learner(n_trajectories, maxiter)

    log_memory("env_config_creation")

    # create demonstrator
    demo = demonstrator.ContinuousDemonstrator(
        env,
        demonstrator_name="Box2dDemonstrator",
        training_algorithm=demo_training_algorithm,
        T=T,
        n_trajectories=n_trajectories,
        solver=MDPSolverApproximationExpectation(
            experiment_name=experiment_name,
            training_algorithm=demo_training_algorithm,
            T=T,
            compute_variance=True,
            policy_config=demo_policy_config,
            training_timesteps=training_timesteps,
        ),
        hugging_face_repo=repo_id,
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
    exit(0)
    # create agent that also matches variances
    agent_variance = learner.ApproximateLearner(
        env,
        demo.mu_demonstrator,
        config_default_learner,
        agent_name="AgentVariance",
        solver=MDPSolverApproximationVariance(
            experiment_name=experiment_name,
            training_algorithm=agent_training_algorithm,
            T=T,
            compute_variance=True,
            policy_config=agent_policy_config,
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
            experiment_name=experiment_name,
            training_algorithm=agent_training_algorithm,
            T=T,
            compute_variance=False,
            policy_config=agent_policy_config,
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
