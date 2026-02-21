import agents.learner as learner
import agents.demonstrator as demonstrator
from environments.object_world_environment import ObjectWorldEnvironment
import MDP_solver_exact as MDPSolver
import numpy as np
import torch
from torch.optim import Adam, RMSprop, SGD, Adamax
from torch.optim.lr_scheduler import LambdaLR, CyclicLR, ReduceLROnPlateau
import wandb
from pathlib import Path
import multiprocessing


def create_objectworld_env(
    gamma=1.0, grid_size=10, T=50, random_start=False, continuous=False
):
    config_env = {
        "theta": [1.0, 1.0, -2.0],
        "gamma": gamma,
        "grid_size": grid_size,
        "random_start": random_start,
        "continuous": continuous,
        "T": T,
        "n_objects": int(2 * grid_size),
    }
    return ObjectWorldEnvironment(config_env)


def create_config_learner():
    config_default_learner = {
        "tol_exp": 0.005,
        "tol_var": 0.5,
        "miniter": 1,
        "maxiter": 1500,
    }

    return config_default_learner


def train():
    # 1. Initialize wandb for the sweep agent
    with wandb.init() as run:
        config = run.config

        # 2. Environment Setup
        env = create_objectworld_env(
            gamma=1.0,
            T=config.T,
            grid_size=config.grid_size,
            random_start=False,
            continuous=False,
        )

        # 3. Create Demonstrator
        demo = demonstrator.ObjectWorldDemonstrator(
            env,
            demonstrator_name="ObjectWorldDemonstrator",
            T=config.T,
            n_trajectories=config.n_trajectories,
        )

        reward_demonstrator = env.compute_true_reward_for_agent(
            demo, config.n_trajectories, config.T
        )

        wandb.log(
            {
                "demonstrator_expected_value": demo.mu_demonstrator[0],
                "demonstrator_variance": demo.mu_demonstrator[1],
                "demonstrator_reward": reward_demonstrator,
                "theta_*": env.theta_reward,
            }
        )

        # 4. Map Optimizer Types
        opt_map = {"Adam": Adam, "RMSprop": RMSprop, "SGD": SGD, "Adamax": Adamax}
        optimizer_cls = opt_map[config.optimizer_type]

        # 5. Define Schedulers based on sweep LRs
        # Using a decay function similar to your original script but parameterized
        lr_lambda_e = lambda step: max(config.lr_decay_rate ** np.log(step + 1), 0.001)
        lr_lambda_v = lambda step: max(
            config.lr_decay_rate_v ** np.log(step + 1), 0.001
        )

        lr_map = {"Lambda": LambdaLR, "Cyclic": CyclicLR, "Reduce": ReduceLROnPlateau}
        lr_v_args = {
            "Lambda": {"lr_lambda": lr_lambda_v},
            "Cyclic": {
                "base_lr": config.learning_rate_v,
                "max_lr": 0.05,
                "step_size_up": 100,
                "mode": "exp_range",
                "gamma": 0.975,
            },
            "Reduce": {"min_lr": 0.0005},
        }

        config_default_learner = create_config_learner()

        agent_expectation = learner.TabularLearner(
            env,
            demo.mu_demonstrator,
            config_default_learner,
            agent_name="AgentExpectation",
            solver=MDPSolver.MDPSolverExactExpectation(config.T),
            learning_rate_e={
                "scheduler": LambdaLR,
                "scheduler_kwargs": {"lr_lambda": lr_lambda_e},
            },
            optimizer_e=optimizer_cls,
            optimizer_e_kwargs={"lr": config.learning_rate_e},
        )
        iter_expectation, time_expectation = agent_expectation.batch_MCE()

        agent_expectation.compute_and_draw(False, False, 4)
        reward_expectation = env.compute_true_reward_for_agent(
            agent_expectation, config.n_trajectories, config.T
        )
        wandb.log(
            {
                "reward_expectation": reward_expectation,
                "reward_diff_expectation": np.abs(
                    reward_demonstrator - reward_expectation
                ),
                "iterations_expectation": iter_expectation,
                "time_total_expectation": sum(time_expectation),
                "time_avg_per_iter_expectation": np.mean(time_expectation),
            }
        )

        # 6. Initialize Agent
        agent_variance = learner.TabularLearner(
            env,
            demo.mu_demonstrator,
            config_default_learner,
            agent_name="AgentVariance",
            solver=MDPSolver.MDPSolverExactVariance(config.T),
            learning_rate_e={
                "scheduler": LambdaLR,
                "scheduler_kwargs": {"lr_lambda": lr_lambda_e},
            },
            learning_rate_v={
                "scheduler": lr_map[config.lr_scheduler],
                "scheduler_kwargs": lr_v_args[config.lr_scheduler],
            },
            optimizer_e=optimizer_cls,
            optimizer_v=optimizer_cls,
            optimizer_e_kwargs={"lr": config.learning_rate_e},
            optimizer_v_kwargs={
                "lr": config.learning_rate_v,
                "weight_decay": config.weight_decay_v,
            },
        )

        # 7. Run MCE-IRL
        iter_variance, time_variance = agent_variance.batch_MCE(
            alternate_every=config.alternate_every, var_factor=config.var_factor
        )
        agent_variance.compute_and_draw(False, False, 7)

        # 8. Evaluation
        reward_variance = env.compute_true_reward_for_agent(
            agent_variance, config.n_trajectories, config.T
        )

        # 9. Log Metrics
        wandb.log(
            {
                "reward_variance": reward_variance,
                "reward_diff_variance": np.abs(reward_demonstrator - reward_variance),
                "iterations_variance": iter_variance,
                "time_total_variance": sum(time_variance),
                "time_avg_per_iter_variance": np.mean(time_variance),
            }
        )


if __name__ == "__main__":
    # Define the sweep configuration
    sweep_config = {
        "method": "bayes",  # Bayesian optimization to find best params faster
        "metric": {"name": "reward_variance", "goal": "maximize"},
        "parameters": {
            "n_trajectories": {"value": 100},
            "grid_size": {"value": 6},
            "T": {"value": 20},
            "learning_rate_e": {
                "distribution": "log_uniform_values",
                "min": 0.01,
                "max": 0.5,
            },
            "learning_rate_v": {
                "distribution": "log_uniform_values",
                "min": 0.005,
                "max": 0.2,
            },
            "lr_scheduler": {"values": ["Lambda", "Cyclic", "Reduce"]},
            "lr_decay_rate": {"values": [0.9, 0.95, 0.99]},
            "lr_decay_rate_v": {"values": [0.8, 0.85, 0.9, 0.95]},
            "optimizer_type": {"values": ["Adam", "RMSprop", "Adamax", "SGD"]},
            "weight_decay_v": {"values": [0.0, 0.001, 0.005]},
            "alternate_every": {"values": [None, 10, 25, 50]},
            "var_factor": {"values": [2, 3, 4, 5, 6, 7, 8, 9]},
        },
    }

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="mceirl-object-world-tuning")

    # Run the agent
    # wandb.agent(
    #     sweep_id, function=train, count=30
    # )  # 'count' is how many combinations to try

    num_agents = 5 

    processes = []
    for i in range(num_agents):
        # We use a helper to call wandb.agent in a separate process
        p = multiprocessing.Process(
            target=wandb.agent, 
            args=(sweep_id,), 
            kwargs={'function': train, 'count': 10} # Each agent does 5 runs
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
