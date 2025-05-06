import numpy as np
from abc import ABC, abstractmethod
from scipy import sparse
from policy import Policy

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)


class Environment(ABC):
    """
    Base class to generalize access to SimpleEnvironment and Gymnasium environments

    Parameters
    ----------
    env_args: dict[Any]
        environment definition parameters depending on the specific environment used
    """

    def __init__(self, env_args: dict):

        self.gamma = env_args["gamma"]
        self.theta_reward = env_args["theta"]

        # true reward per state
        self.reward = None

    def get_reward_for_given_theta(self, theta_e: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        theta_e : ndarray

        Returns
        -------
        reward : ndarray
        """
        reward = self.feature_matrix.dot(theta_e)
        return np.array(reward)

    def get_variance_for_given_theta(self, theta_v: np.ndarray) -> np.ndarray:
        """
        computes the variance term for soft value iteration for given theta_v

        Parameters
        ----------
        theta_v : ndarray

        Returns
        -------
        variance : ndarray
        """
        variance = [
            (theta_v.dot(self.feature_matrix[i, :])).dot(self.feature_matrix[i, :])
            for i in range(self.feature_matrix.shape[0])
        ]

        return np.array(variance)
    
    @abstractmethod
    def _compute_state_feature_matrix(self):
        pass

    def get_state_feature_matrix(self) -> np.ndarray:
        """
        Getter function for (state) feature matrix
        """
        return self.feature_matrix.copy()

    @abstractmethod
    def _compute_transition_matrix(self):
        pass

    @abstractmethod
    def _get_initial_distribution(self):
        pass

    def _compute_transition_sparse_list(self) -> list[sparse.csc_matrix]:
        """
        Computes the sparse transition list for the different actions

        Returns
        -------
        T_sparse_list : list[sparse.csr_matrix]
        """
        T_sparse_list = []
        for a in range(self.n_actions):
            T_sparse_list.append(sparse.csr_matrix(self.T_matrix[:, :, a]))

        return T_sparse_list

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def render(self, policy:Policy, T: int = 20, store:bool=False,**kwargs):
        pass

    def compute_true_reward_for_agent(self, agent, n_trajectories:int=None, T:int=None) -> float:
        """
        Compute the true reward in the environment either with the given policy or with trajectories

        Parameters
        ----------
        agent
            agent/demonstrator that should be evaluated in the environment
        n_trajectories : int
            number of trajectories to use, if None then we use the reward vector

        Returns
        -------
        reward : float
            true reward for the given policy
        """

        if n_trajectories is None:
            
            a = agent.solver.compute_feature_SVF_bellmann(
                            self, agent.policy.pi
                        )[0]

            return np.dot(
                        self.reward,
                        agent.solver.compute_feature_SVF_bellmann(
                            self, agent.policy.pi
                        )[0],
                    )
        
        else:
            # compute reward based on trajectories
            rewards = []
            for i in range(n_trajectories):
                trajectory = agent.solver.generate_episode(self, agent.policy, T)[0]

                rewards.append(sum([trajectory[j][3]*self.gamma**j for j in range(len(trajectory))]))
            return np.mean(rewards)
