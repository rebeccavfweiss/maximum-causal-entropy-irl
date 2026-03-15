"""
Utilities for saving and loading pre-trained ObjectWorld demonstrators.

A demonstrator is uniquely identified by its environment configuration
(grid_size, objects, theta, random_start, continuous) and demonstration
parameters (T, n_trajectories).  We hash these into a deterministic key
and persist policy + mu_demonstrator as .npz files.
"""

import hashlib
import json
import numpy as np
from pathlib import Path

from agents.demonstrator import ObjectWorldDemonstrator
from environments.object_world_environment import ObjectWorldEnvironment
from policy import TabularPolicy
from solvers.MDP_solver_exact import MDPSolverExactExpectation

DEFAULT_CACHE_DIR = Path("experiments/object_world/demonstrators")


def _build_key_dict(
    grid_size: int,
    objects_config: list[dict],
    theta: list[float],
    random_start: bool,
    continuous: bool,
    demo_T: int,
    n_trajectories: int | None,
) -> dict:
    """Build a canonical dict that uniquely identifies a demonstrator."""
    return {
        "grid_size": grid_size,
        "objects": sorted(objects_config, key=lambda o: (o["x"], o["y"])),
        "theta": theta,
        "random_start": random_start,
        "continuous": continuous,
        "demo_T": demo_T,
        "n_trajectories": n_trajectories,
    }


def compute_cache_key(
    grid_size: int,
    objects_config: list[dict],
    theta: list[float],
    random_start: bool,
    continuous: bool,
    demo_T: int,
    n_trajectories: int | None,
) -> str:
    """Return a deterministic hex digest for the given demonstrator config."""
    key_dict = _build_key_dict(
        grid_size, objects_config, theta, random_start, continuous,
        demo_T, n_trajectories,
    )
    key_str = json.dumps(key_dict, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def save_demonstrator(
    demo: ObjectWorldDemonstrator,
    objects_config: list[dict],
    cache_dir: Path = DEFAULT_CACHE_DIR,
    grid_size: int = None,
    theta: list[float] = None,
    random_start: bool = False,
    continuous: bool = False,
) -> Path:
    """Save demonstrator policy and mu_demonstrator to disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    gs = grid_size if grid_size is not None else demo.env.grid_size
    th = theta if theta is not None else list(demo.env.theta_reward)

    key = compute_cache_key(
        gs, objects_config, th, random_start, continuous,
        demo.T, demo.n_trajectories,
    )

    save_path = cache_dir / f"demo_{key}.npz"
    mu_exp, mu_var = demo.mu_demonstrator

    np.savez(
        save_path,
        policy_pi=demo.policy.pi,
        mu_expectation=mu_exp,
        mu_variance=mu_var,
        V=demo.V,
    )

    # Also save metadata for human inspection
    meta_path = cache_dir / f"demo_{key}_meta.json"
    meta = _build_key_dict(
        gs, objects_config, th, random_start, continuous,
        demo.T, demo.n_trajectories,
    )
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return save_path


def load_demonstrator(
    env: ObjectWorldEnvironment,
    objects_config: list[dict],
    demo_T: int,
    n_trajectories: int | None,
    theta: list[float] = None,
    random_start: bool = False,
    continuous: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> ObjectWorldDemonstrator | None:
    """
    Load a cached demonstrator if it exists.

    Returns a fully reconstructed ObjectWorldDemonstrator with policy and
    mu_demonstrator restored, or None if no cache entry is found.
    """
    th = theta if theta is not None else list(env.theta_reward)

    key = compute_cache_key(
        env.grid_size, objects_config, th, random_start, continuous,
        demo_T, n_trajectories,
    )

    save_path = cache_dir / f"demo_{key}.npz"
    if not save_path.exists():
        return None

    data = np.load(save_path)

    # Reconstruct the demonstrator without running value iteration
    demo = ObjectWorldDemonstrator.__new__(ObjectWorldDemonstrator)
    # Manually set all attributes that __init__ would set
    demo.env = env
    demo.agent_name = "ObjectWorldDemonstrator"
    demo.T = demo_T
    demo.n_trajectories = n_trajectories
    demo.solver = MDPSolverExactExpectation(demo_T, compute_variance=True)
    demo.policy = TabularPolicy(data["policy_pi"])
    demo.mu_demonstrator = (data["mu_expectation"], data["mu_variance"])
    demo.V = data["V"]

    return demo


def load_or_train_demonstrator(
    env: ObjectWorldEnvironment,
    objects_config: list[dict],
    demo_T: int,
    n_trajectories: int | None = None,
    theta: list[float] = None,
    random_start: bool = False,
    continuous: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    save_if_trained: bool = True,
) -> ObjectWorldDemonstrator:
    """
    Load a cached demonstrator or train a new one.

    If a cached version exists, loads and returns it.
    Otherwise, trains a new demonstrator, optionally saves it, and returns it.
    """
    demo = load_demonstrator(
        env, objects_config, demo_T, n_trajectories,
        theta=theta, random_start=random_start, continuous=continuous,
        cache_dir=cache_dir,
    )
    if demo is not None:
        print(f"Loaded cached demonstrator from {cache_dir}")
        return demo

    print("No cached demonstrator found, training new one...")
    demo = ObjectWorldDemonstrator(
        env,
        demonstrator_name="ObjectWorldDemonstrator",
        T=demo_T,
        n_trajectories=n_trajectories,
    )

    if save_if_trained:
        path = save_demonstrator(
            demo, objects_config,
            cache_dir=cache_dir,
            grid_size=env.grid_size,
            theta=theta,
            random_start=random_start,
            continuous=continuous,
        )
        print(f"Saved demonstrator to {path}")

    return demo
