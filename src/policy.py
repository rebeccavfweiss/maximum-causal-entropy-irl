from abc import ABC, abstractmethod
import numpy as np

class Policy(ABC):
    """
    Wrapper class for differnt kinds of policies to offer the same interface to other parts of the algorithm no matter what the policy looks like.

    """
    @abstractmethod
    def predict(obs, t:int=None) -> int:
        pass


class TabularPolicy(Policy):
    """
    Specific implementation of the Policy interface to use a tablular form of a policy

    Parameters
    ----------
    pi : nd.array
        tabluar policy to use
    """

    def __init__(self, pi:np.ndarray):
        self.pi = pi

    def predict(self, obs, t:int=None) -> int:
        """
        Predicts an action given an observation in the environment

        Parameters
        ----------
        obs
            observation from the environment
        t : int
            time step (only relevant if policy time dependent)

        Returns
        -------
        action : int
            action to take in the given state
        """

        probs = self.pi[t, obs]
        action = int(np.random.choice(np.arange(len(probs)), p=probs))

        return action