import numpy as np
from abc import ABC, abstractmethod
from scipy import sparse

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

        # variables that need to be defined in the specific environment
        self.n_states = None
        self.n_actions  = None
        self.n_features = None

        self.T = None
        self.T_sparse_list = None
        self.terminat_states = None
        self.feature_matrix = None

        #true reward per state
        self.reward = None
        self.InitD = None


    def get_reward_for_given_theta(self, theta_e:np.ndarray) -> np.ndarray:
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

    def get_variance_for_given_theta(self, theta_v:np.ndarray) -> np.ndarray:
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

    def get_state_feature_matrix(self) -> np.ndarray:
        """
        Getter function for (state) feature matrix
        """
        return self.feature_matrix.copy()

    @abstractmethod
    def _compute_transition_matrix(self):
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
            T_sparse_list.append(sparse.csr_matrix(self.T[:, :, a]))

        return T_sparse_list
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def step(self, action):
        pass
    
    @abstractmethod
    def render(
        self,
        pi,
        T: int = 20,
        **kwargs
    ):
        pass