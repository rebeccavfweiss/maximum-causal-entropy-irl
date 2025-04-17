import MDPSolver
from environment import Environment
from simple_environment import SimpleEnvironment
from gymnasium_environment import GymEnvironment
from abc import ABC, abstractmethod
import copy
import numpy as np
import random
from tqdm import tqdm


class Demonstrator(ABC):
    """
    Abstract demonstrator class to generalize demonstrators in different environments

    Parameters
    ----------
    env : environment.Environment
        the environment representing the setting of the problem
    demonstrator_name : str
        name of the demonstrator
    T : int
        finite horizon value for the MDP solver
    gamma : float
        reward discount factor
    """

    def __init__(
        self, env: Environment, demonstrator_name: str, T: int = 45, gamma: float = 1.0
    ):
        self.V = None
        self.pi = None
        self.reward = None
        self.env = env
        self.demonstrator_name = demonstrator_name
        self.T = T
        self.gamma = gamma
        self.solver = MDPSolver.MDPSolverExpectation(T, compute_variance=True)
        self.mu_demonstrator = None

    @abstractmethod
    def _define_policy(self):
        pass

    def get_mu_using_reward_features(self) -> tuple[np.ndarray, np.ndarray]:
        """
        computes feature expectation and variance terms for the demonstrator using a predefined policy and computing the value function based on it

        Returns
        -------
        feature expectation and variance arrays
        """

        _, mu, nu = self.solver.compute_feature_SVF_bellmann_averaged(self.env, self.pi)

        return (
            mu,
            nu,
        )

    def _compute_value_function(self, pi: np.ndarray) -> np.ndarray:
        """
        Computes the state-dependent value function based on the given policy

        Parameters
        ----------
        pi : ndarray
            policy to use

        Returns
        -------
        ndarray
            matrix containing the time-dependent value function
        """
        V = np.zeros((self.T + 1, self.env.n_states))

        for t in range(self.T - 1, -1, -1):
            for s in range(self.env.n_states):
                V[t, s] = sum(
                    pi[t, s, a]
                    * sum(
                        self.env.T_matrix[s, s_prime, a]
                        * (self.env.reward[s] + self.env.gamma * V[t + 1, s_prime])
                        for s_prime in range(self.env.n_states)
                    )
                    for a in range(self.env.n_actions)
                )

        return V

    def draw(self, show: bool = False, store: bool = False, fignum: int = 0) -> None:
        """
        draws the policy of the demonstrator as long as it has been computed before, else a warning is thrown

        Parameters
        ----------
        show : bool
            whether or not the plot should be shown
        store : bool
            whether or not the plot should be stored
        fignum : int
            identifier number for the figure
        """

        self.reward = copy.deepcopy(self.env.reward)
        self.env.render(
            V=self.V,
            pi=self.pi,
            reward=self.reward,
            show=show,
            strname=self.demonstrator_name,
            fignum=fignum,
            store=store,
            T=self.T,
        )


class SimpleDemonstrator(Demonstrator):
    """
    class to implement the demonstrator that computes its policy for a certain reward using value iteration
    policy is hard coded

    Parameters
    ----------
    env : environment.Environment
        the environment representing the setting of the problem
    demonstrator_name : str
        name of the demonstrator
    T : int
        finite horizon value for the MDP solver
    gamma : float
        reward discount factor
    """

    def __init__(
        self,
        env: SimpleEnvironment,
        demonstrator_name: str,
        T: int = 45,
        gamma: float = 1.0,
    ):
        super().__init__(env, demonstrator_name, T, gamma)

        self.pi = self._define_policy()

        self.mu_demonstrator = self.get_mu_using_reward_features()

    def _define_policy(self):
        """
        Method to compute or manually define the policy of the demonstrator

        Returns
        -------
        pi_s : numpy.ndarray
            policy of the demonstrator
        """
        pi_s = np.zeros((self.T, self.env.n_states, self.env.n_actions))

        # define the demonstrator's behavior in a specific way
        for t in range(pi_s.shape[0]):
            pi_s[t, self.env.point_to_int(4, 0), self.env.actions["up"]] = 0.5
            pi_s[t, self.env.point_to_int(4, 0), self.env.actions["down"]] = 0.5

            pi_s[t, self.env.point_to_int(3, 0), self.env.actions["up"]] = 1.0
            pi_s[t, self.env.point_to_int(2, 0), self.env.actions["up"]] = 1.0
            pi_s[t, self.env.point_to_int(1, 0), self.env.actions["up"]] = 1.0

            pi_s[t, self.env.point_to_int(5, 0), self.env.actions["down"]] = 1.0
            pi_s[t, self.env.point_to_int(6, 0), self.env.actions["down"]] = 1.0
            pi_s[t, self.env.point_to_int(7, 0), self.env.actions["down"]] = 1.0

        self.V = self._compute_value_function(pi_s)

        return pi_s


class GymDemonstrator(Demonstrator):
    """
    Demonstrator in a given Gymnasium environment. Currently this demonstrator will be just trained using normal Q-learning.

    Parameters
    ----------
    env : environment.Environment
        the environment representing the setting of the problem
    demonstrator_name : str
        name of the demonstrator
    T : int
        finite horizon value for the MDP solver
    gamma : float
        reward discount factor
    learning_rate : float
        learning rate for training
    n_training_episodes : int
        number of episodes for training
    """

    def __init__(
        self,
        env: GymEnvironment,
        demonstrator_name: str,
        T: int = 45,
        gamma: float = 1.0,
    ):
        super().__init__(env, demonstrator_name, T, gamma)

        self.pi = self._define_policy()
        self.mu_demonstrator = self.get_mu_using_reward_features()

    def _define_policy(self):
        # TODO generalize to more than one hole (for now assumption only one)
        pi_s = np.zeros((self.T, self.env.n_states, self.env.n_actions))

        lava_states = self.env.env.forbidden_states

        if lava_states[0][0] == lava_states[1][0]:
            # x coordinate of the lava are the same -> vertical line
            missing_y = [lava_states[i][1]!= i+1 for i in range(self.env.height-3)]
            # add one to get the correct coordinate (due to borders)
            hole_y = np.argmax(missing_y) + 1

            state = [1,1,0]
            state_index = self.env.env.to_state_index(*state)
            #orient downward
            pi_s[0,state_index,1] = 1.0
            
            for y in range(1,hole_y):
                state = [1, y, 1]
                state_index = self.env.env.to_state_index(*state)
                # move forward in these states
                pi_s[0,state_index, 2] = 1.0

            state = [1,hole_y,1]
            state_index = self.env.env.to_state_index(*state)
            #orient to the right
            pi_s[0,state_index,0] = 1.0

            for x in range(1, self.env.width-2):
                state = [x, hole_y, 0]
                state_index = self.env.env.to_state_index(*state)
                # move forward in these states
                pi_s[0,state_index, 2] = 1.0

            state = [self.env.width-2,hole_y,0]
            state_index = self.env.env.to_state_index(*state)
            #orient downward
            pi_s[0,state_index,1] = 1.0

            for y in range(hole_y, self.env.height-2):
                state = [self.env.width-2, y, 1]
                state_index = self.env.env.to_state_index(*state)
                # move forward in these states
                pi_s[0,state_index, 2] = 1.0   

        else:
            # y coordinate of the lava are the same -> horizontal line
            missing_x = [lava_states[i][0]!= i+1 for i in range(self.env.height-3)]
            # add one to get the correct coordinate (due to borders)
            hole_x = np.argmax(missing_x) + 1
            
            for x in range(1,hole_x):
                state = [x, 1, 0]
                state_index = self.env.env.to_state_index(*state)
                # move forward in these states
                pi_s[0,state_index, 2] = 1.0

            state = [hole_x,1,0]
            state_index = self.env.env.to_state_index(*state)
            #orient downward
            pi_s[0,state_index,1] = 1.0

            for y in range(1, self.env.height-2):
                state = [hole_x, y, 1]
                state_index = self.env.env.to_state_index(*state)
                # move forward in these states
                pi_s[0,state_index, 2] = 1.0

            state = [hole_x,self.env.height-2, 1]
            state_index = self.env.env.to_state_index(*state)
            #orient to the right
            pi_s[0,state_index,0] = 1.0

            for x in range(hole_x, self.env.width-2):
                state = [x, self.env.height-2, 0]
                state_index = self.env.env.to_state_index(*state)
                # move forward in these states
                pi_s[0,state_index, 2] = 1.0  

        for t in range(1,self.T):
            #time independent policy
            pi_s[t] = pi_s[0]

        return pi_s
