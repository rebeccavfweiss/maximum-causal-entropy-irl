import agents.learner as learner
import agents.demonstrator as demonstrator
from environments.car_racing_environment import CarRacingEnvironment
from MDP_solver_approximation import (
    MDPSolverApproximationExpectation,
    MDPSolverApproximationVariance,
)
import numpy as np
import psutil
import gc
import wandb
from utils import laplacian_2d
from scipy.sparse import kron, eye


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
        "tol": 0.0005,
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


def compute_heuristic_theta_e(width, height):
    # Create a mask that favors the center
    y = np.linspace(-1, 1, width)
    x = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dist_from_center = np.sqrt(X**2 + Y**2) - 0.5 * X

    # Invert to give high weight to center, low to edges
    center_mask = 1.0 - dist_from_center  # range ~[0, 1]
    center_mask = np.clip(center_mask, 0, 0.9)

    # Normalize
    # center_mask /= center_mask.sum()

    return -np.tile(center_mask.flatten(), 4)


def temporal_diff_matrix(num_frames, frame_size):
    # block tridiagonal matrix with I, -I on off-diagonals
    I = np.eye(frame_size)
    D = np.zeros((num_frames * frame_size, num_frames * frame_size))
    for i in range(num_frames - 1):
        D[
            i * frame_size : (i + 1) * frame_size,
            i * frame_size : (i + 1) * frame_size,
        ] += I
        D[
            (i + 1) * frame_size : (i + 2) * frame_size,
            (i + 1) * frame_size : (i + 2) * frame_size,
        ] += I
        D[
            i * frame_size : (i + 1) * frame_size,
            (i + 1) * frame_size : (i + 2) * frame_size,
        ] -= I
        D[
            (i + 1) * frame_size : (i + 2) * frame_size,
            i * frame_size : (i + 1) * frame_size,
        ] -= I
    return D


def compute_heuristic_theta_v(h_theta_e, alpha, beta, gamma):
    Q_track = np.diag(h_theta_e)

    L = laplacian_2d(env.frame_height, env.frame_width, env.n_frames)
    I4 = eye(4)
    Q_spatial = kron(I4, L)  # shape (4*84*84, 4*84*84)

    D = temporal_diff_matrix(env.n_frames, env.frame_height * env.frame_width)

    return alpha * Q_track + beta * Q_spatial + gamma * D


if __name__ == "__main__":

    show = False
    store = True
    continuous_actions = True
    experiment_name = "car_racing" + ("_continuous" if continuous_actions else "_discrete")

    maxiter = 20
    n_trajectories = 10
    training_timesteps = 10000
    policy_config = dict(buffer_size=50000, tau=0.005, gamma=1.0, train_freq=3)
    # does not really change anything so for now just limit T (i.e. technically goal of the agents now to just survive on the track as long as possible until time runs out as will not be possible to achieve lap in restricted time)
    lap_percent_complete = 0.95
    T = 200

    weight_track_adherence = 1.0  # track adherence
    weight_spatial_smoothness = 0.1  # spatial smoothness
    weight_temporal_consistency = 0.05  # temporal consistency (small)

    learning_rate = lambda step: max(0.95 ** (step + 1), 0.01)

    wandb.init(
        project="mceirl-car-racing",
        name=f"{experiment_name}-iter{maxiter}-sac_iter{training_timesteps}-T{T}-traj{n_trajectories}-adh{weight_track_adherence}-smooth{weight_spatial_smoothness}-tempcons{weight_temporal_consistency}",
        config={
            "maxiter": maxiter,
            "n_trajectories": n_trajectories,
            "sac_dqn_timesteps": training_timesteps,
            "sac_dqn_buffer_size": policy_config["buffer_size"],
            "lap_percent_complete": lap_percent_complete,
            "T": T,
            "track_adherence": weight_track_adherence,
            "spatial_smoothness": weight_spatial_smoothness,
            "temporal_consistency": weight_temporal_consistency,
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

    heuristic_theta_e = compute_heuristic_theta_e(env.frame_width, env.frame_height)
    heuristic_theta_v = compute_heuristic_theta_v(
        heuristic_theta_e,
        weight_track_adherence,
        weight_spatial_smoothness,
        weight_temporal_consistency,
    )

    # Learner config
    config_default_learner = create_config_learner(n_trajectories, maxiter)

    log_memory("env_config_creation")

    # create demonstrator
    demo = demonstrator.CarRacingDemonstrator(
        env,
        demonstrator_name="CarRacingDemonstrator",
        continuous_actions=continuous_actions,
        T=T,
        n_trajectories=n_trajectories,
        solver=MDPSolverApproximationExpectation(
            experiment_name=experiment_name,
            continuous_actions=continuous_actions,
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

    demo.render(show, store, 0)

    # clean up
    del demo.policy
    log_memory("demonstrator_policy_cleanup")

    # create agent that uses only expectation matching
    agent_expectation = learner.ApproximateLearner(
        env,
        demo.mu_demonstrator,
        config_default_learner,
        agent_name="AgentExpectation",
        solver=MDPSolverApproximationExpectation(
            experiment_name=experiment_name,
            continuous_actions=continuous_actions,
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

    # create agent that also matches variances
    agent_variance = learner.ApproximateLearner(
        env,
        demo.mu_demonstrator,
        config_default_learner,
        agent_name="AgentVariance",
        solver=MDPSolverApproximationVariance(
            experiment_name=experiment_name,
            continuous_actions=continuous_actions,
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

    wandb.finish()
