import agents.learner as learner
import agents.demonstrator as demonstrator
from environments.car_racing_environment import CarRacingEnvironment
from solvers.MDP_solver_approximation import (
    MDPSolverApproximationExpectation,
    MDPSolverApproximationVariance,
)
import numpy as np
import psutil
import gc
import wandb


def create_carracing_env(
    lap_complete_percent: float = 0.95,
    T: int = 1000,
    gamma: float = 0.99,
    continuous_actions: bool = True,
):

    config_env = {
        "gamma": gamma,
        "lap_complete_percent": lap_complete_percent,
        "T": T,
        "n_frames": 4,
        "width": 84,
        "height": 84,
        "continuous_actions": continuous_actions,
    }

    env = CarRacingEnvironment(config_env)
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


def temporal_diff_matrix(num_frames: int, frame_size: int):
    """
    Construct a quadratic penalty matrix D such that:
    R(s) = s^T D s = sum_i ||f_i - f_{i-1}||^2

    Args:
        num_frames (int): number of stacked frames (e.g., 4)
        frame_size (int): number of pixels per frame (e.g., 84*84 = 7056)

    Returns:
        D (np.ndarray): block matrix of shape (num_frames*frame_size, num_frames*frame_size)
    """
    size = num_frames * frame_size
    D = np.zeros((size, size))

    I = np.eye(frame_size)

    for i in range(1, num_frames):
        # Indices for frame i-1 and i
        a = (i - 1) * frame_size
        b = i * frame_size

        # Expand (f_i - f_{i-1})^T (f_i - f_{i-1})
        # = f_i^T f_i + f_{i-1}^T f_{i-1} - 2 f_i^T f_{i-1}
        D[a : a + frame_size, a : a + frame_size] += I
        D[b : b + frame_size, b : b + frame_size] += I
        D[a : a + frame_size, b : b + frame_size] -= I
        D[b : b + frame_size, a : a + frame_size] -= I

    return D


def compute_heuristics(width, height, weight_forward, weight_speed_limitation):
    # Create a mask that favors the center
    y = np.linspace(-1, 1, width)
    x = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dist_from_center = np.sqrt(X**2 + Y**2) - 0.5 * X

    # Invert to give high weight to center, low to edges
    center_mask = 1.0 - dist_from_center  # range ~[0, 1]
    center_mask = np.clip(center_mask, 0, 0.9)

    # Normalize
    center_mask /= 255.0  # center_mask.sum()

    D = temporal_diff_matrix(4, width * height)

    h_theta_e = -np.tile(center_mask.flatten(), 4)
    h_theta_v = (
        np.diag(h_theta_e) + weight_forward * D - weight_speed_limitation * D.dot(D)
    )

    return h_theta_e, h_theta_v


if __name__ == "__main__":

    show = False
    store = True
    continuous_actions = False
    experiment_name = "car_racing" + (
        "_continuous" if continuous_actions else "_discrete"
    )

    demo_training_algorithm = "ppo" if continuous_actions else "dqn"
    agent_training_algorithm = "sac" if continuous_actions else "dqn"

    maxiter = 50
    n_trajectories = 150
    training_timesteps = 350000
    policy_config = dict(
        buffer_size=50000, tau=0.005, gamma=1.0, train_freq=5, device="auto"
    )
    # does not really change anything so for now just limit T (i.e. technically goal of the agents now to just survive on the track as long as possible until time runs out as will not be possible to achieve lap in restricted time)
    lap_percent_complete = 0.33
    T = 400
    weight_forward = 0.1
    weight_speed_limitation = 0.05

    learning_rate = lambda step: max(0.975 ** (step + 1), 0.01)

    wandb.init(
        project="mceirl-car-racing",
        name=f"{experiment_name}-iter{maxiter}-sac_iter{training_timesteps}-T{T}-traj{n_trajectories}",
        config={
            "maxiter": maxiter,
            "n_trajectories": n_trajectories,
            "sac_dqn_timesteps": training_timesteps,
            "sac_dqn_buffer_size": policy_config["buffer_size"],
            "lap_percent_complete": lap_percent_complete,
            "weight_forward": weight_forward,
            "weight_speed_limitation": weight_speed_limitation,
            "T": T,
            "actions_space": (
                "continous action space"
                if continuous_actions
                else "discrete action space"
            ),
        },
    )

    log_memory("start")

    # create the environment
    env = create_carracing_env(
        lap_complete_percent=lap_percent_complete,
        T=T,
        gamma=policy_config["gamma"],
        continuous_actions=continuous_actions,
    )

    heuristic_theta_e, heuristic_theta_v = compute_heuristics(
        env.frame_width, env.frame_height, weight_forward, weight_speed_limitation
    )

    # Learner config
    config_default_learner = create_config_learner(n_trajectories, maxiter)

    log_memory("env_config_creation")

    # create demonstrator
    demo = demonstrator.ContinuousDemonstrator(
        env,
        demonstrator_name="CarRacingDemonstrator",
        training_algorithm=demo_training_algorithm,
        T=T,
        n_trajectories=n_trajectories,
        solver=MDPSolverApproximationExpectation(
            experiment_name=experiment_name,
            training_algorithm=demo_training_algorithm,
            T=T,
            compute_variance=True,
            policy_config=policy_config,
            training_timesteps=training_timesteps,
        ),
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
            experiment_name=experiment_name,
            training_algorithm=agent_training_algorithm,
            T=T,
            compute_variance=True,
            policy_config=policy_config,
            training_timesteps=training_timesteps,
        ),
        learning_rate=learning_rate,
        heuristic_theta_e=heuristic_theta_e,
        heuristic_theta_v=heuristic_theta_v,
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
            policy_config=policy_config,
            training_timesteps=training_timesteps,
        ),
        learning_rate=learning_rate,
        heuristic_theta_e=heuristic_theta_e,
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
