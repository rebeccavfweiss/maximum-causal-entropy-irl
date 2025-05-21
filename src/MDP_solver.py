import numpy as np
from abc import ABC, abstractmethod
from environments.environment import Environment
from policy import Policy

np.set_printoptions(suppress=True)
np.set_printoptions(precision=12)
np.set_printoptions(linewidth=500)


class MDPSolver(ABC):
    """
    Interface for an MDP solver collecting all necessary function interfaces for the two different kinds of MDP solvers that we implement.

    Parameters
    ----------
    T : int
        finite horizon value
    compute_variance : bool
            whether or not variance term should be computed (for efficiency reasons will only be computed if necessary)

    """

    def __init__(self, T: int, compute_variance: bool):
        self.T = T
        self.compute_variance = compute_variance

    @abstractmethod
    def soft_value_iteration(env: Environment, values: dict[str:any]):
        pass

    def generate_episode(
        self,
        env: Environment,
        policy: Policy,
        len_episode: int,
    ) -> tuple[list[tuple[any, int, any, float]], np.ndarray]:
        """
        generates an episode in the given setting

        Parameters
        ----------
        env : environment.Environment
            the environment representing the setting of the problem
        policy : ndarray
            policy to use (must be stochastic)
        len_episode : int
            episode length

        Returns
        -------
        episode : list[tuple[any, int, any, float]]
            list of current state, action, next state, and reward
        """

        # Selecting a start state according to InitD
        state = env.reset()

        episode = []
        for t in range(len_episode):
            if ((env.terminal_states is not None) and (state in env.terminal_states)) or (t == self.T):
                break
            action = policy.predict(state, t)
            next_state, reward, _, _ = env.step(action)
            episode.append((state, action, next_state, reward))
            state = next_state

        return episode

    def compute_feature_SVF_bellmann_averaged(
        self,
        env: Environment,
        policy: Policy,
        n_trajectories: int = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        computes average feature SVF

        Parameters
        ----------
        env : environment.Environment
            the environment representing the setting of the problem
        policy : ndarray
            policy to use (time dependent)
        trajectories : list
            trajectories to use if policy doesnt exist

        Returns
        -------
        mean feature expectation
        mean feature variance
        """

        # To ensure stochastic behaviour in the feature expectation and state-visitation frequencies (run num_iter times)
        if n_trajectories is not None and n_trajectories > 0:
            num_iter = n_trajectories
        else:
            #use policy directly
            num_iter = 1

        mu_avg = None
        nu_avg = None

        for i in range(num_iter):
            if n_trajectories is None or n_trajectories == 0:
                trajectory = None
            else:
                trajectory = self.generate_episode(env, policy, self.T)

            feature_expectation, feature_variance = (
                self.compute_feature_SVF_bellmann(env, policy, trajectory)
            )

            if mu_avg is None:
                mu_avg = np.array(feature_expectation, dtype=np.float64)
                nu_avg = np.array(feature_variance, dtype=np.float64)
            else:
                mu_avg += (np.array(feature_expectation) - mu_avg) / (i + 1)
                nu_avg += (np.array(feature_variance) - nu_avg) / (i + 1)

        return mu_avg, nu_avg

    @abstractmethod
    def compute_feature_SVF_bellmann(
        self,
        env: Environment,
        policy: Policy,
        trajectory: list[tuple[any, int, any, float]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        pass

    def compute_value_function_bellmann_averaged(
        self, env: Environment, policy, values: dict[str:any], num_iter: int = None
    ) -> np.ndarray:
        # for most environments this function is not needed but it will still be called for a uniform procedure no matter what setting we use
        return None
