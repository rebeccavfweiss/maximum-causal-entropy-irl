"""
Shared training pipeline for hyperparameter tuning sweeps.

Extracts the repeated experiment logic into reusable functions
that each per-environment sweep script delegates to.
"""

import copy
import gc
import yaml
import numpy as np
import psutil
import wandb
from pathlib import Path
from torch.optim import Adam, RMSprop, SGD, Adamax
from torch.optim.lr_scheduler import LambdaLR, CyclicLR, ReduceLROnPlateau

import agents.learner as learner
import solvers.MDP_solver_exact as MDPSolverExact
from solvers.MDP_solver_approximation import (
    MDPSolverApproximationExpectation,
    MDPSolverApproximationVariance,
)

OPTIMIZER_MAP = {
    "Adam": Adam,
    "RMSprop": RMSprop,
    "SGD": SGD,
    "Adamax": Adamax,
}

SCHEDULER_MAP = {
    "Lambda": LambdaLR,
    "Cyclic": CyclicLR,
    "Reduce": ReduceLROnPlateau,
}

# Parameters that only apply to the variance agent
VARIANCE_ONLY_PARAMS = {
    "tol_var",
    "optimizer_type_v",
    "lr_v",
    "weight_decay_v",
    "lr_scheduler_v",
    "lr_decay_rate_v",
    "alternate_every",
    "var_factor",
}


def load_config(yaml_path: str) -> dict:
    """Load and return the full YAML config as a dict."""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def prepare_sweep_config(sweep_config_from_yaml: dict, agent_type: str) -> dict:
    """
    Adjust the raw sweep config from YAML based on agent_type.

    For "expectation": sets metric to reward_expectation, removes
    variance-only params from the search space.
    For "variance": sets metric to reward_variance, keeps all params.
    """
    config = copy.deepcopy(sweep_config_from_yaml)

    if agent_type == "expectation":
        config["metric"] = {"name": "reward_expectation", "goal": "maximize"}
        for param in VARIANCE_ONLY_PARAMS:
            config["parameters"].pop(param, None)
    else:
        config["metric"] = {"name": "reward_variance", "goal": "maximize"}

    return config


def build_optimizer_config(sweep_config, agent_type: str) -> dict:
    """
    Build optimizer/scheduler dicts from sweep params for Learner.__init__().

    Uses optimizer_type_e for theta_e and optimizer_type_v for theta_v
    (they can be different optimizer types).
    """
    opt_cls_e = OPTIMIZER_MAP[sweep_config.optimizer_type_e]
    decay_rate_e = getattr(sweep_config, "lr_decay_rate", 0.95)

    lr_lambda_e = lambda step, dr=decay_rate_e: max(
        dr ** np.log(step + 1), 0.001
    )

    sched_name_e = getattr(sweep_config, "lr_scheduler_e", "Lambda")

    lr_e_scheduler_args = {
        "Lambda": {"lr_lambda": lr_lambda_e},
        "Cyclic": {
            "base_lr": sweep_config.lr_e,
            "max_lr": 0.05,
            "step_size_up": 100,
            "mode": "exp_range",
            "gamma": 0.975,
        },
        "Reduce": {"min_lr": 0.0005},
    }

    result = {
        "optimizer_e": opt_cls_e,
        "optimizer_e_kwargs": {
            "lr": sweep_config.lr_e,
            "weight_decay": getattr(sweep_config, "weight_decay_e", 0.0),
        },
        "learning_rate_e": {
            "scheduler": SCHEDULER_MAP[sched_name_e],
            "scheduler_kwargs": lr_e_scheduler_args[sched_name_e],
        },
    }

    if agent_type == "variance":
        opt_cls_v = OPTIMIZER_MAP[sweep_config.optimizer_type_v]
        decay_rate_v = getattr(sweep_config, "lr_decay_rate_v", decay_rate_e)

        lr_lambda_v = lambda step, dr=decay_rate_v: max(
            dr ** np.log(step + 1), 0.001
        )

        sched_name_v = getattr(sweep_config, "lr_scheduler_v", "Lambda")

        lr_v_scheduler_args = {
            "Lambda": {"lr_lambda": lr_lambda_v},
            "Cyclic": {
                "base_lr": sweep_config.lr_v,
                "max_lr": 0.05,
                "step_size_up": 100,
                "mode": "exp_range",
                "gamma": 0.975,
            },
            "Reduce": {"min_lr": 0.0005},
        }

        result["optimizer_v"] = opt_cls_v
        result["optimizer_v_kwargs"] = {
            "lr": sweep_config.lr_v,
            "weight_decay": getattr(sweep_config, "weight_decay_v", 0.0),
        }
        result["learning_rate_v"] = {
            "scheduler": SCHEDULER_MAP[sched_name_v],
            "scheduler_kwargs": lr_v_scheduler_args[sched_name_v],
        }

    return result


def build_learner_config(sweep_config, agent_type: str) -> dict:
    """Build the config_agent dict for Learner from sweep params."""
    config = {
        "tol_exp": sweep_config.tol_exp,
        "maxiter": sweep_config.maxiter,
        "miniter": getattr(sweep_config, "miniter", 1),
    }

    if hasattr(sweep_config, "n_trajectories"):
        config["n_trajectories"] = sweep_config.n_trajectories

    if agent_type == "variance":
        config["tol_var"] = sweep_config.tol_var

    return config


def build_policy_config(
    sweep_config, features_extractor_class=None
) -> tuple[dict, dict]:
    """
    Build policy_config and policy_kwargs for approximate solvers.

    Returns (policy_config, policy_kwargs).
    """
    policy_type = getattr(sweep_config, "policy_type", "MlpPolicy")

    policy_config = {
        "policy": policy_type,
        "buffer_size": getattr(sweep_config, "buffer_size", 50000),
        "tau": getattr(sweep_config, "tau", 0.005),
        "gamma": getattr(sweep_config, "gamma_rl", 1.0),
        "train_freq": getattr(sweep_config, "train_freq", 5),
        "device": "auto",
    }

    policy_kwargs = None
    if features_extractor_class is not None:
        features_dim = getattr(sweep_config, "features_dim", 128)
        policy_kwargs = {
            "features_extractor_class": features_extractor_class,
            "features_extractor_kwargs": {"features_dim": features_dim},
        }
    elif hasattr(sweep_config, "net_arch"):
        net_arch = sweep_config.net_arch
        if isinstance(net_arch, (list, tuple)):
            policy_kwargs = {"net_arch": list(net_arch)}

    return policy_config, policy_kwargs


def log_demonstrator_metrics(env, demo, n_trajectories: int, T: int) -> float:
    """Evaluate the demonstrator's true reward, log to wandb, return reward."""
    reward_demonstrator = env.compute_true_reward_for_agent(
        demo, n_trajectories, T
    )
    wandb.log(
        {
            "demonstrator_expected_value": demo.mu_demonstrator[0],
            "demonstrator_variance": demo.mu_demonstrator[1],
            "demonstrator_reward": reward_demonstrator,
        }
    )
    return reward_demonstrator


def log_memory(stage: str = "") -> None:
    """Log system memory usage to wandb."""
    mem = psutil.virtual_memory()
    wandb.log(
        {
            f"memory_free_mb_{stage}": mem.available / (1024**2),
            f"memory_used_percent_{stage}": mem.percent,
            f"memory_total_mb_{stage}": mem.total / (1024**2),
        }
    )


def train_and_evaluate_tabular(
    env,
    demo,
    agent_type: str,
    learner_config: dict,
    optimizer_config: dict,
    T: int,
    n_trajectories_eval: int,
    alternate_every=None,
    var_factor: int = 2,
    show: bool = False,
    store: bool = False,
) -> None:
    """
    Train and evaluate a single tabular agent (expectation or variance).

    Logs reward, iterations, and timing metrics to the active wandb run.
    """
    reward_demonstrator = log_demonstrator_metrics(
        env, demo, n_trajectories_eval, T
    )

    if agent_type == "expectation":
        agent = learner.TabularLearner(
            env,
            demo.mu_demonstrator,
            learner_config,
            agent_name="AgentExpectation",
            solver=MDPSolverExact.MDPSolverExactExpectation(T),
            learning_rate_e=optimizer_config["learning_rate_e"],
            optimizer_e=optimizer_config["optimizer_e"],
            optimizer_e_kwargs=optimizer_config["optimizer_e_kwargs"],
        )
        iters, times = agent.batch_MCE()
        agent.compute_and_draw(show, store, 4)
        reward = env.compute_true_reward_for_agent(
            agent, n_trajectories_eval, T
        )
        wandb.log(
            {
                "reward_expectation": reward,
                "reward_diff_expectation": np.abs(reward_demonstrator - reward),
                "iterations_expectation": iters,
                "time_total_expectation": sum(times),
                "time_avg_per_iter_expectation": np.mean(times),
            }
        )
    else:
        agent = learner.TabularLearner(
            env,
            demo.mu_demonstrator,
            learner_config,
            agent_name="AgentVariance",
            solver=MDPSolverExact.MDPSolverExactVariance(T),
            learning_rate_e=optimizer_config["learning_rate_e"],
            learning_rate_v=optimizer_config["learning_rate_v"],
            optimizer_e=optimizer_config["optimizer_e"],
            optimizer_v=optimizer_config["optimizer_v"],
            optimizer_e_kwargs=optimizer_config["optimizer_e_kwargs"],
            optimizer_v_kwargs=optimizer_config["optimizer_v_kwargs"],
        )
        iters, times = agent.batch_MCE(
            alternate_every=alternate_every, var_factor=var_factor
        )
        agent.compute_and_draw(show, store, 7)
        reward = env.compute_true_reward_for_agent(
            agent, n_trajectories_eval, T
        )
        wandb.log(
            {
                "reward_variance": reward,
                "reward_diff_variance": np.abs(reward_demonstrator - reward),
                "iterations_variance": iters,
                "time_total_variance": sum(times),
                "time_avg_per_iter_variance": np.mean(times),
            }
        )


def train_and_evaluate_approximate(
    env,
    demo,
    agent_type: str,
    learner_config: dict,
    optimizer_config: dict,
    policy_config: dict,
    policy_kwargs: dict,
    experiment_name: str,
    training_algorithm: str,
    training_timesteps: int,
    T: int,
    n_trajectories_eval: int,
    alternate_every=None,
    var_factor: int = 2,
    heuristic_theta_e=None,
    heuristic_theta_v=None,
    show: bool = False,
    store: bool = True,
) -> None:
    """
    Train and evaluate a single approximate agent (expectation or variance).

    Includes model artifact logging, memory monitoring, and cleanup.
    """
    log_memory("start")

    reward_demonstrator = log_demonstrator_metrics(
        env, demo, n_trajectories_eval, T
    )

    # Render demonstrator video
    path_to_file = demo.render(show, store, 0)
    if path_to_file is not None:
        wandb.log(
            {
                f"eval/video_{demo.agent_name}": wandb.Video(
                    path_to_file, fps=4, format="mp4"
                )
            }
        )

    # Clean up demonstrator policy
    if hasattr(demo, "policy") and demo.policy is not None:
        del demo.policy
    log_memory("demonstrator_policy_cleanup")

    if agent_type == "expectation":
        agent_name = "AgentExpectation"
        solver = MDPSolverApproximationExpectation(
            experiment_name=experiment_name,
            training_algorithm=training_algorithm,
            T=T,
            compute_variance=False,
            policy_config=policy_config,
            policy_kwargs=policy_kwargs,
            training_timesteps=training_timesteps,
        )
        agent = learner.ApproximateLearner(
            env,
            demo.mu_demonstrator,
            learner_config,
            agent_name=agent_name,
            solver=solver,
            learning_rate_e=optimizer_config["learning_rate_e"],
            optimizer_e=optimizer_config["optimizer_e"],
            optimizer_e_kwargs=optimizer_config["optimizer_e_kwargs"],
            heuristic_theta_e=heuristic_theta_e,
        )
        iters, times = agent.batch_MCE()
        agent.compute_and_draw(show, store, 2)
        reward = env.compute_true_reward_for_agent(
            agent, n_trajectories_eval, T
        )
        log_memory("agent_expectation_finished")

        artifact = wandb.Artifact("agent_expectation_model", type="model")
        artifact.add_file(agent.solver.model_dir / "best_model.zip")
        wandb.log_artifact(artifact)

        wandb.log(
            {
                "reward_expectation": reward,
                "reward_diff_expectation": np.abs(
                    reward_demonstrator - reward
                ),
                "iterations_expectation": iters,
                "time_total_expectation": sum(times),
                "time_avg_per_iter_expectation": np.mean(times),
            }
        )
    else:
        agent_name = "AgentVariance"
        solver = MDPSolverApproximationVariance(
            experiment_name=experiment_name,
            training_algorithm=training_algorithm,
            T=T,
            compute_variance=True,
            policy_config=policy_config,
            policy_kwargs=policy_kwargs,
            training_timesteps=training_timesteps,
        )
        agent = learner.ApproximateLearner(
            env,
            demo.mu_demonstrator,
            learner_config,
            agent_name=agent_name,
            solver=solver,
            learning_rate_e=optimizer_config["learning_rate_e"],
            learning_rate_v=optimizer_config.get("learning_rate_v"),
            optimizer_e=optimizer_config["optimizer_e"],
            optimizer_v=optimizer_config.get("optimizer_v"),
            optimizer_e_kwargs=optimizer_config["optimizer_e_kwargs"],
            optimizer_v_kwargs=optimizer_config.get("optimizer_v_kwargs"),
            heuristic_theta_e=heuristic_theta_e,
            heuristic_theta_v=heuristic_theta_v,
        )
        iters, times = agent.batch_MCE(
            alternate_every=alternate_every, var_factor=var_factor
        )
        agent.compute_and_draw(show, store, 4)
        reward = env.compute_true_reward_for_agent(
            agent, n_trajectories_eval, T
        )
        log_memory("agent_variance_finished")

        artifact = wandb.Artifact("agent_variance_model", type="model")
        artifact.add_file(agent.solver.model_dir / "best_model.zip")
        wandb.log_artifact(artifact)

        wandb.log(
            {
                "reward_variance": reward,
                "reward_diff_variance": np.abs(
                    reward_demonstrator - reward
                ),
                "iterations_variance": iters,
                "time_total_variance": sum(times),
                "time_avg_per_iter_variance": np.mean(times),
            }
        )

    # Cleanup
    if hasattr(agent, "policy") and agent.policy is not None:
        del agent.policy
    del agent
    gc.collect()
    log_memory("agent_cleanup")
