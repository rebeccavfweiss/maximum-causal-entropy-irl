import agents.learner as learner
import agents.demonstrator as demonstrator
from environments.car_racing_environment import CarRacingEnvironment
from MDP_solver_approximation import MDPSolverApproximationExpectation, MDPSolverApproximationVariance
import numpy as np
import psutil
import gc
import wandb

def create_carracing_env(lap_complete_percent:float=0.95, T:int = 1000, gamma:float = 0.99):

    config_env = {
        "gamma": gamma,
        "lap_complete_percent": lap_complete_percent,
        "T":T,
        "n_frames" : 4,
        "width": 84,
        "height": 84,
    }

    env = CarRacingEnvironment(config_env)
    return env


def create_config_learner(n_trajectories:int = 1, maxiter:int=3):
    config_default_learner = {"tol": 0.0005, "miniter": 1, "maxiter": maxiter, "n_trajectories":n_trajectories}

    return config_default_learner

def log_memory(stage=""):
    mem = psutil.virtual_memory()
    wandb.log({
        f"memory_free_mb_{stage}": mem.available / (1024 ** 2),
        f"memory_used_percent_{stage}": mem.percent,
        f"memory_total_mb_{stage}": mem.total / (1024 ** 2),
    })

if __name__ == "__main__":

    show = False
    store = True
    experiment_name = "car_racing"

    maxiter = 30
    n_trajectories = 100
    sac_timesteps = 7500
    sac_buffer_size = 50000
    sac_tau = 0.05
    sac_gamma = 0.95
    # does not really change anything so for now just limit T (i.e. technically goal of the agents now to just survive on the track as long as possible until time runs out as will not be possible to achieve lap in restricted time)
    lap_percent_complete=0.95
    T = 200

    learning_rate = lambda step: max(0.99 ** (step + 1), 0.01)


    wandb.init(
    project="mceirl-car-racing",
    name=f"{experiment_name}-run",
    config={
        "maxiter": maxiter,
        "n_trajectories": n_trajectories,
        "sac_timesteps": sac_timesteps,
        "sac_buffer_size": sac_buffer_size,
        "lap_percent_complete": lap_percent_complete,
        "T": T
    }
)

    log_memory("start")

    # create the environment
    env = create_carracing_env(lap_complete_percent=lap_percent_complete, T=T, gamma=sac_gamma)

    # Learner config
    config_default_learner = create_config_learner(n_trajectories, maxiter)

    log_memory("env_config_creation")

    # create demonstrator
    demo = demonstrator.CarRacingDemonstrator(
        env,
        demonstrator_name="CarRacingDemonstrator",
        T=T,
        n_trajectories=n_trajectories,
        solver = MDPSolverApproximationExpectation(experiment_name, T,True, sac_timesteps, sac_buffer_size)
    )

    log_memory("demonstrator_creation")

    reward_demonstrator = env.compute_true_reward_for_agent(demo, n_trajectories, T)

    wandb.log({"demonstrator_expected_value" : demo.mu_demonstrator[0],
            "demonstrator_variance": demo.mu_demonstrator[1],
            "demonstrator_reward": reward_demonstrator})

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
        solver=MDPSolverApproximationExpectation(experiment_name, T, sac_timesteps=sac_timesteps, sac_buffer_size=sac_buffer_size),
        learning_rate= learning_rate
    )

    iter_expectation, time_expectation = agent_expectation.batch_MCE()
    agent_expectation.compute_and_draw(show, store, 2)
    reward_expectation = env.compute_true_reward_for_agent(agent_expectation, n_trajectories, T)

    log_memory("agent_expectation_finished")   

    artifact = wandb.Artifact(f"agent_expectation_model", type="model")
    artifact.add_file(agent_expectation.solver.model_dir/"best_model.zip")
    wandb.log_artifact(artifact) 


    wandb.log({
        "reward_expectation": reward_expectation,
        "reward_diff_expectation": np.abs(reward_demonstrator - reward_expectation),
        "iterations_expectation": iter_expectation,
        "time_total_expectation": sum(time_expectation),
        "time_avg_per_iter_expectation": np.mean(time_expectation),
    })

    del agent_expectation
    gc.collect()
    log_memory("agent_expectation_policy_cleanup")

    # create agent that also matches variances
    agent_variance = learner.ApproximateLearner(
        env,
        demo.mu_demonstrator,
        config_default_learner,
        agent_name="AgentVariance",
        solver=MDPSolverApproximationVariance(experiment_name,T, sac_timesteps=sac_timesteps, sac_buffer_size=sac_buffer_size),
        learning_rate=learning_rate
    )

    iter_variance, time_variance = agent_variance.batch_MCE()
    agent_variance.compute_and_draw(show, store, 4)
    reward_variance = env.compute_true_reward_for_agent(agent_variance, n_trajectories, T)

    log_memory("agent_variance_finished")

    artifact = wandb.Artifact(f"agent_variance_model", type="model")
    artifact.add_file(agent_variance.solver.model_dir/ "best_model.zip")
    wandb.log_artifact(artifact) 

    # Free up memory
    del agent_variance.policy
    log_memory("agent_expectation_policy_cleanup")

    wandb.log({
        "reward_variance": reward_variance,
        "reward_diff_variance": np.abs(reward_demonstrator - reward_variance),
        "iterations_variance": iter_variance,
        "time_total_variance": sum(time_variance),
        "time_avg_per_iter_variance": np.mean(time_variance),
    })

    wandb.finish()

