import MDPSolver
from environment import Environment
from simple_environment import SimpleEnvironment
from gymnasium_environment import GymEnvironment
from abc import ABC
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
                        self.env.T[s, s_prime, a]
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
        self.env.draw(
            self.V,
            self.pi,
            self.reward,
            show,
            self.demonstrator_name,
            fignum,
            store,
            self.T,
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
        self.pi = pi_s

        self.mu_demonstrator = self.get_mu_using_reward_features()


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
        learning_rate: float = 0.05,
        n_training_episodes: int = 5000,
    ):
        super().__init__(env, demonstrator_name, T, gamma)

        self.learning_rate = learning_rate
        self.n_training_episodes = n_training_episodes

        self.pi = self.__train_demonstrator()
        self.mu_demonstrator = self.get_mu_using_reward_features()

    def __train_demonstrator(self) -> np.ndarray:
        """
        Training method using Q-Learning to train the Demonstrator in the given environment

        Returns
        -------
        pi_s : ndarray
            computed policy based on Q table
        """

        pi_s = np.zeros((self.T, self.env.n_states, self.env.n_actions))
        Qtable = np.zeros((self.env.n_states, self.env.n_actions))

        max_epsilon = 1.0
        min_epsilon = 0.05
        decay_rate = 0.0005

        for episode in tqdm(range(self.n_training_episodes)):
            # Reduce epsilon (because we need less and less exploration)
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
                -decay_rate * episode
            )
            state = self.env.reset()
            terminated = False
            truncated = False

            for step in range(self.T):
                # Choose the action At using epsilon greedy policy
                action = self.__epsilon_greedy_policy(Qtable, state, epsilon)

                # Take action At and observe Rt+1 and St+1
                new_state, reward, terminated, truncated = self.env.step(action)

                Qtable[state][action] = Qtable[state][action] + self.learning_rate * (
                    reward
                    + self.gamma * np.max(Qtable[new_state])
                    - Qtable[state][action]
                )

                if terminated or truncated:
                    break

                state = new_state

        # define (deterministic) policy based on Q table
        for state in range(self.env.n_states):
            for t in range(self.T):
                best_action = np.argmax(Qtable[state])
                pi_s[:, state, best_action] = 1.0  # Best action with probability 1

        return pi_s

    def __epsilon_greedy_policy(
        self, Qtable: np.ndarray, state: int, epsilon: float
    ) -> int:
        """
        Helper function to compute an epsilon greedy policy based on the Q table

        Parameters
        ----------
        Qtable : ndarray
            Q table to base policy on
        state : int
            current state
        epsilon : float
            exploration probability

        Returns
        -------
        action : int
            what action to chose in the given state
        """
        random_num = random.uniform(0, 1)

        if random_num > epsilon:
            action = self.__greedy_policy(Qtable, state)
        else:
            action = self.env.action_sample()

        return action

    def __greedy_policy(self, Qtable, state) -> int:
        """
        Helper function to compute a greedy policy based on the Q table

        Parameters
        ----------
        Qtable : ndarray
            Q table to base policy on
        state : int
            current state

        Returns
        -------
        action : int
            what action to chose in the given state
        """
        action = np.argmax(Qtable[state][:])

        return action
