"""
JAX-based Box2D experiment script.

Uses JaxSolver + JaxApproximateLearner with warm-starting instead of
SB3-based ApproximateLearner. Supports DQN (discrete) and SAC (continuous).
Optionally runs an MMD learner as a third comparison agent.
"""

import agents.demonstrator as demonstrator
from agents.jax_learner import JaxApproximateLearner
from agents.mmd_learner import MMDLearner
from environments.box2d_jax_environment import Box2DJaxEnvironment
from solvers.MDP_solver_jax import JaxSolverExpectation, JaxSolverVariance
from solvers.MDP_solver_approximation import MDPSolverApproximationExpectation
import numpy as np
import optax
import psutil
import gc
import wandb
import torch


def create_box2d_jax_env(
    env_id: str = "LunarLander-v3",
    T: int = 1000,
    gamma: float = 0.99,
    continuous: bool = False,
):
    config_env = {
        "env_id": env_id,
        "gamma": gamma,
        "T": T,
        "continuous": continuous,
    }
    return Box2DJaxEnvironment(config_env)


def create_config_learner(n_trajectories: int = 500, maxiter: int = 500):
    return {
        "tol_exp": 0.0005,
        "tol_var": 0.0125,
        "miniter": 1,
        "maxiter": maxiter,
        "n_trajectories": n_trajectories,
    }


def log_memory(stage=""):
    mem = psutil.virtual_memory()
    wandb.log(
        {
            f"memory_free_mb_{stage}": mem.available / (1024**2),
            f"memory_used_percent_{stage}": mem.percent,
        }
    )


if __name__ == "__main__":

    show = False
    store = True
    continuous = False
    env_id = "LunarLander-v3"
    run_mmd = False  # Set to True to also run MMD learner

    experiment_name = env_id + ("_continuous_jax" if continuous else "_discrete_jax")
    training_algorithm = "sac" if continuous else "dqn"

    maxiter = 500
    n_trajectories = 500
    full_training_timesteps = 200_000
    finetune_timesteps = 5_000
    T = 800 if env_id == "LunarLander-v3" else 300

    training_config = {
        "gamma": 0.99,
        "lr": 1e-3,
        "buffer_size": 50000,
        "batch_size": 64,
        "hidden_dims": (64, 64),
        "tau": 0.005,
        "train_freq": 4,
        "eval_freq": 5000,
    }

    wandb.init(
        project=f"mceirl-{env_id}-jax",
        name=f"{experiment_name}-iter{maxiter}-T{T}-traj{n_trajectories}",
        config={
            "maxiter": maxiter,
            "n_trajectories": n_trajectories,
            "full_training_timesteps": full_training_timesteps,
            "finetune_timesteps": finetune_timesteps,
            "T": T,
            "training_algorithm": training_algorithm,
            "warm_starting": True,
        },
    )

    log_memory("start")

    # Create environment (JAX-compatible, no SB3 wrappers)
    env = create_box2d_jax_env(
        env_id=env_id,
        T=T,
        gamma=0.99,
        continuous=continuous,
    )

    config_learner = create_config_learner(n_trajectories, maxiter)

    log_memory("env_config_creation")

    # Create demonstrator (still uses SB3 PPO, as it only produces mu_demonstrator)
    # We need the SB3-based environment for the demonstrator
    from environments.box2d_environment import Box2DEnvironment

    demo_env = Box2DEnvironment({
        "env_id": env_id,
        "gamma": 0.99,
        "T": T,
        "continuous": continuous,
        "enable_wind": False,
    })

    demo_policy_config = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[256, 256],
        gamma=1.0,
    )

    demo = demonstrator.ContinuousDemonstrator(
        demo_env,
        demonstrator_name="Box2dDemonstrator",
        training_algorithm="ppo",
        T=T,
        n_trajectories=n_trajectories,
        solver=MDPSolverApproximationExpectation(
            experiment_name=experiment_name,
            training_algorithm="ppo",
            T=T,
            compute_variance=True,
            policy_config=demo_policy_config,
            training_timesteps=500_000,
        ),
        hugging_face_repo="Chiz" if env_id == "LunarLander-v3" else "matamaki",
    )

    log_memory("demonstrator_creation")

    reward_demonstrator = demo_env.compute_true_reward_for_agent(demo, n_trajectories, T)
    wandb.log({
        "demonstrator_expected_value": demo.mu_demonstrator[0],
        "demonstrator_variance": demo.mu_demonstrator[1],
        "demonstrator_reward": reward_demonstrator,
    })

    # Clean up demonstrator policy
    del demo.policy
    log_memory("demonstrator_policy_cleanup")

    # ---- Agent: Expectation (JAX + warm-starting) ----
    solver_exp = JaxSolverExpectation(
        experiment_name=experiment_name,
        training_algorithm=training_algorithm,
        training_config=training_config,
        T=T,
        compute_variance=False,
        full_training_timesteps=full_training_timesteps,
        finetune_timesteps=finetune_timesteps,
    )

    agent_expectation = JaxApproximateLearner(
        env,
        demo.mu_demonstrator,
        config_learner,
        agent_name="AgentExpectation_JAX",
        solver=solver_exp,
        lr_e=0.1,
    )

    iter_exp, time_exp = agent_expectation.batch_MCE()
    reward_expectation = env.compute_true_reward_for_agent(
        agent_expectation, n_trajectories, T
    )

    wandb.log({
        "reward_expectation": reward_expectation,
        "reward_diff_expectation": np.abs(reward_demonstrator - reward_expectation),
        "iterations_expectation": iter_exp,
        "time_total_expectation": sum(time_exp),
        "time_avg_per_iter_expectation": np.mean(time_exp),
    })

    del agent_expectation
    gc.collect()
    log_memory("agent_expectation_cleanup")

    # ---- Agent: Variance (JAX + warm-starting) ----
    solver_var = JaxSolverVariance(
        experiment_name=experiment_name,
        training_algorithm=training_algorithm,
        training_config=training_config,
        T=T,
        compute_variance=True,
        full_training_timesteps=full_training_timesteps,
        finetune_timesteps=finetune_timesteps,
    )

    agent_variance = JaxApproximateLearner(
        env,
        demo.mu_demonstrator,
        config_learner,
        agent_name="AgentVariance_JAX",
        solver=solver_var,
        lr_e=0.1,
        lr_v=0.05,
    )

    iter_var, time_var = agent_variance.batch_MCE()
    reward_variance = env.compute_true_reward_for_agent(
        agent_variance, n_trajectories, T
    )

    wandb.log({
        "reward_variance": reward_variance,
        "reward_diff_variance": np.abs(reward_demonstrator - reward_variance),
        "iterations_variance": iter_var,
        "time_total_variance": sum(time_var),
        "time_avg_per_iter_variance": np.mean(time_var),
    })

    del agent_variance
    gc.collect()
    log_memory("agent_variance_cleanup")

    # ---- Agent: MMD (optional) ----
    if run_mmd:
        # Collect expert trajectory features
        # Re-create demonstrator policy for feature collection
        demo_for_features = demonstrator.ContinuousDemonstrator(
            demo_env,
            demonstrator_name="Box2dDemonstrator",
            training_algorithm="ppo",
            T=T,
            n_trajectories=n_trajectories,
            solver=MDPSolverApproximationExpectation(
                experiment_name=experiment_name,
                training_algorithm="ppo",
                T=T,
                compute_variance=True,
                policy_config=demo_policy_config,
                training_timesteps=500_000,
            ),
            hugging_face_repo="Chiz" if env_id == "LunarLander-v3" else "matamaki",
        )

        solver_mmd = JaxSolverExpectation(
            experiment_name=experiment_name,
            training_algorithm=training_algorithm,
            training_config=training_config,
            T=T,
            compute_variance=False,
            full_training_timesteps=full_training_timesteps,
            finetune_timesteps=finetune_timesteps,
        )

        # Collect expert features
        expert_features = []
        for _ in range(n_trajectories):
            trajectory = solver_mmd.generate_episode(
                env, demo_for_features.policy, T
            )
            feat_sum = trajectory[0][0].flatten().astype(np.float32)
            for i in range(len(trajectory)):
                feat_sum += env.gamma ** (i + 1) * trajectory[i][2].flatten().astype(np.float32)
            expert_features.append(feat_sum)
        expert_features = np.array(expert_features)

        del demo_for_features
        gc.collect()

        agent_mmd = MMDLearner(
            env,
            demo.mu_demonstrator,
            config_learner,
            agent_name="AgentMMD_JAX",
            solver=solver_mmd,
            expert_features=expert_features,
            tol_mmd=0.01,
        )

        iter_mmd, time_mmd = agent_mmd.batch_MCE()
        reward_mmd = env.compute_true_reward_for_agent(agent_mmd, n_trajectories, T)

        wandb.log({
            "reward_mmd": reward_mmd,
            "reward_diff_mmd": np.abs(reward_demonstrator - reward_mmd),
            "iterations_mmd": iter_mmd,
            "time_total_mmd": sum(time_mmd),
        })

        del agent_mmd
        gc.collect()

    wandb.finish()
