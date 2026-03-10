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

# Parameters that only apply to the MMD agent
MMD_ONLY_PARAMS = {
    "tol_mmd",
    "kernel_bandwidth",
}

# Parameters that only apply to expectation/variance (not MMD)
MOMENT_MATCHING_ONLY_PARAMS = {
    "tol_exp",
    "optimizer_type_e",
    "lr_e",
    "weight_decay_e",
    "lr_scheduler_e",
    "lr_decay_rate",
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

    For "expectation": sets metric to reward_diff_expectation, removes
    variance-only and MMD-only params from the search space.
    For "variance": sets metric to reward_diff_variance, removes MMD-only params.
    For "mmd": sets metric to reward_diff_mmd, removes moment-matching-only params.
    """
    config = copy.deepcopy(sweep_config_from_yaml)

    if agent_type == "expectation":
        config["metric"] = {"name": "reward_diff_expectation", "goal": "maximize"}
        for param in VARIANCE_ONLY_PARAMS | MMD_ONLY_PARAMS:
            config["parameters"].pop(param, None)
    elif agent_type == "variance":
        config["metric"] = {"name": "reward_diff_variance", "goal": "maximize"}
        for param in MMD_ONLY_PARAMS:
            config["parameters"].pop(param, None)
    elif agent_type == "mmd":
        config["metric"] = {"name": "reward_diff_mmd", "goal": "maximize"}
        for param in MOMENT_MATCHING_ONLY_PARAMS:
            config["parameters"].pop(param, None)

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
            "max_lr": sweep_config.lr_e +0.05,
            "step_size_up": 100,
            "mode": "exp_range",
            "gamma": decay_rate_e,
        },
        "Reduce": {"min_lr": 0.0001, "factor": 0.5},
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
                "max_lr": sweep_config.lr_v + 0.05,
                "step_size_up": 100,
                "mode": "exp_range",
                "gamma": decay_rate_v,
            },
            "Reduce": {"min_lr": 0.0001, "factor": 0.5},
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
    if agent_type == "mmd":
        config = {
            "tol_exp": 1.0,  # not used by MMD, but required by parent
            "maxiter": sweep_config.maxiter,
            "miniter": getattr(sweep_config, "miniter", 1),
        }
    else:
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
                "reward_diff_expectation": reward - reward_demonstrator,
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
                "reward_diff_variance": reward - reward_demonstrator,
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
                "reward_diff_variance": reward - 
                    reward_demonstrator,
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


def train_and_evaluate_jax(
    env,
    demo_env,
    demo,
    agent_type: str,
    learner_config: dict,
    training_config: dict,
    experiment_name: str,
    training_algorithm: str,
    full_training_timesteps: int,
    finetune_timesteps: int,
    T: int,
    n_trajectories_eval: int,
    lr_e: float = 0.1,
    lr_v: float = 0.05,
    lr_decay_rate_e: float = 0.95,
    lr_decay_rate_v: float = 0.9,
    alternate_every=None,
    var_factor: int = 2,
    show: bool = False,
    store: bool = True,
) -> None:
    """
    Train and evaluate a JAX-based agent with warm-starting.

    Uses JaxApproximateLearner + JaxSolver instead of SB3-based equivalents.
    """
    from agents.jax_learner import JaxApproximateLearner
    from solvers.MDP_solver_jax import JaxSolverExpectation, JaxSolverVariance

    log_memory("start")

    reward_demonstrator = log_demonstrator_metrics(
        demo_env, demo, n_trajectories_eval, T
    )

    # Clean up demonstrator policy
    if hasattr(demo, "policy") and demo.policy is not None:
        del demo.policy
    log_memory("demonstrator_policy_cleanup")

    if agent_type == "expectation":
        agent_name = "AgentExpectation_JAX"
        solver = JaxSolverExpectation(
            experiment_name=experiment_name,
            training_algorithm=training_algorithm,
            training_config=training_config,
            T=T,
            compute_variance=False,
            full_training_timesteps=full_training_timesteps,
            finetune_timesteps=finetune_timesteps,
        )
        agent = JaxApproximateLearner(
            env,
            demo.mu_demonstrator,
            learner_config,
            agent_name=agent_name,
            solver=solver,
            lr_e=lr_e,
            lr_decay_rate_e=lr_decay_rate_e,
        )
        iters, times = agent.batch_MCE()
        reward = env.compute_true_reward_for_agent(
            agent, n_trajectories_eval, T
        )
        log_memory("agent_expectation_finished")

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
        agent_name = "AgentVariance_JAX"
        solver = JaxSolverVariance(
            experiment_name=experiment_name,
            training_algorithm=training_algorithm,
            training_config=training_config,
            T=T,
            compute_variance=True,
            full_training_timesteps=full_training_timesteps,
            finetune_timesteps=finetune_timesteps,
        )
        agent = JaxApproximateLearner(
            env,
            demo.mu_demonstrator,
            learner_config,
            agent_name=agent_name,
            solver=solver,
            lr_e=lr_e,
            lr_v=lr_v,
            lr_decay_rate_e=lr_decay_rate_e,
            lr_decay_rate_v=lr_decay_rate_v,
        )
        iters, times = agent.batch_MCE(
            alternate_every=alternate_every, var_factor=var_factor
        )
        reward = env.compute_true_reward_for_agent(
            agent, n_trajectories_eval, T
        )
        log_memory("agent_variance_finished")

        wandb.log(
            {
                "reward_variance": reward,
                "reward_diff_variance": reward - 
                    reward_demonstrator,
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
    log_memory("agent_jax_cleanup")


def train_and_evaluate_mmd(
    env,
    demo_env,
    demo,
    learner_config: dict,
    training_config: dict,
    experiment_name: str,
    training_algorithm: str,
    full_training_timesteps: int,
    finetune_timesteps: int,
    T: int,
    n_trajectories_eval: int,
    kernel_bandwidth: float = None,
    tol_mmd: float = 0.01,
    show: bool = False,
    store: bool = True,
) -> None:
    """
    Train and evaluate an MMD-based JAX agent.

    Collects expert trajectory features from the demonstrator, then uses
    MMDLearner with the MMD witness function as reward.
    """
    from agents.mmd_learner import MMDLearner
    from solvers.MDP_solver_jax import JaxSolverExpectation

    log_memory("start")

    reward_demonstrator = log_demonstrator_metrics(
        demo_env, demo, n_trajectories_eval, T
    )

    # Collect expert trajectory features for MMD
    expert_solver = JaxSolverExpectation(
        experiment_name=experiment_name + "_expert_features",
        training_algorithm=training_algorithm,
        training_config=training_config,
        T=T,
        compute_variance=False,
        full_training_timesteps=full_training_timesteps,
        finetune_timesteps=finetune_timesteps,
    )
    n_traj = learner_config.get("n_trajectories", 100)
    expert_features = []
    for _ in range(n_traj):
        trajectory = expert_solver.generate_episode(env, demo.policy, T)
        if len(trajectory) == 0:
            expert_features.append(
                np.zeros(env.n_features, dtype=np.float32)
            )
            continue
        feat_sum = trajectory[0][0].flatten().astype(np.float32)
        for i in range(len(trajectory)):
            feat_sum += (
                env.gamma ** (i + 1)
                * trajectory[i][2].flatten().astype(np.float32)
            )
        expert_features.append(feat_sum)
    expert_features = np.array(expert_features)

    # Clean up demonstrator policy
    if hasattr(demo, "policy") and demo.policy is not None:
        del demo.policy
    log_memory("demonstrator_policy_cleanup")

    agent_name = "AgentMMD_JAX"
    solver = JaxSolverExpectation(
        experiment_name=experiment_name,
        training_algorithm=training_algorithm,
        training_config=training_config,
        T=T,
        compute_variance=False,
        full_training_timesteps=full_training_timesteps,
        finetune_timesteps=finetune_timesteps,
    )
    agent = MMDLearner(
        env,
        demo.mu_demonstrator,
        learner_config,
        agent_name=agent_name,
        solver=solver,
        expert_features=expert_features,
        kernel_bandwidth=kernel_bandwidth,
        tol_mmd=tol_mmd,
    )
    iters, times = agent.batch_MCE()
    reward = env.compute_true_reward_for_agent(
        agent, n_trajectories_eval, T
    )
    log_memory("agent_mmd_finished")

    wandb.log(
        {
            "reward_mmd": reward,
            "reward_diff_mmd": reward - reward_demonstrator,
            "iterations_mmd": iters,
            "time_total_mmd": sum(times),
            "time_avg_per_iter_mmd": np.mean(times),
        }
    )

    # Cleanup
    if hasattr(agent, "policy") and agent.policy is not None:
        del agent.policy
    del agent
    gc.collect()
    log_memory("agent_mmd_cleanup")


def train_and_evaluate_tabular_mmd(
    env,
    demo,
    learner_config: dict,
    T: int,
    n_trajectories_eval: int,
    kernel_bandwidth: float = None,
    tol_mmd: float = 0.01,
    show: bool = False,
    store: bool = False,
) -> None:
    """
    Train and evaluate a tabular MMD agent using exact MDP solvers.

    Collects expert trajectory features from the demonstrator, then uses
    TabularMMDLearner with the MMD witness function as per-state reward.
    """
    from agents.tabular_mmd_learner import TabularMMDLearner

    reward_demonstrator = log_demonstrator_metrics(
        env, demo, n_trajectories_eval, T
    )

    # Collect expert trajectory features
    expert_solver = MDPSolverExact.MDPSolverExactExpectation(T)
    n_traj = learner_config.get("n_trajectories", 100)
    feature_matrix = env.get_state_feature_matrix()
    expert_features = []
    for _ in range(n_traj):
        trajectory = expert_solver.generate_episode(env, demo.policy, T)
        if len(trajectory) == 0:
            expert_features.append(
                np.zeros(env.n_features, dtype=np.float32)
            )
            continue
        feat_sum = feature_matrix[trajectory[0][0]].astype(np.float32)
        for i in range(len(trajectory)):
            feat_sum += (
                env.gamma ** (i + 1)
                * feature_matrix[trajectory[i][2]].astype(np.float32)
            )
        expert_features.append(feat_sum)
    expert_features = np.array(expert_features)

    agent_name = "AgentMMD_Tabular"
    solver = MDPSolverExact.MDPSolverExactExpectation(T)
    agent = TabularMMDLearner(
        env,
        demo.mu_demonstrator,
        learner_config,
        agent_name=agent_name,
        solver=solver,
        expert_features=expert_features,
        kernel_bandwidth=kernel_bandwidth,
        tol_mmd=tol_mmd,
    )
    iters, times = agent.batch_MCE()
    agent.compute_and_draw(show, store, 10)
    reward = env.compute_true_reward_for_agent(
        agent, n_trajectories_eval, T
    )

    wandb.log(
        {
            "reward_mmd": reward,
            "reward_diff_mmd": reward - reward_demonstrator,
            "iterations_mmd": iters,
            "time_total_mmd": sum(times),
            "time_avg_per_iter_mmd": np.mean(times),
        }
    )
