import MDPSolver
from environments.environment import Environment
from environments.simple_environment import SimpleEnvironment
from environments.minigrid_environment import MinigridEnvironment
from policy import TabularPolicy
from abc import ABC, abstractmethod
from agents.agent import Agent
import copy
import numpy as np
from operator import itemgetter


class Demonstrator(Agent):
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
    n_trajectories : int
        number of trajectories to use to compute the expectation values
    """

    def __init__(
        self,
        env: Environment,
        demonstrator_name: str,
        T: int = 45,
        n_trajectories: int = None,
    ):  
        super().__init__(env, demonstrator_name)
        self.T = T
        self.V = None
        self.policy = None
        self.trajectories = None
        self.n_trajectories = n_trajectories
        self.reward = None
        self.solver = MDPSolver.MDPSolverExpectation(T, compute_variance=True)
        self.mu_demonstrator = None
        self.reward = copy.deepcopy(self.env.reward)

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
        # TODO make MDP solver work with policies not the raw tables
        _, mu, nu = self.solver.compute_feature_SVF_bellmann_averaged(
            self.env, self.policy.pi, self.trajectories
        )

        return (
            mu,
            nu,
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
    n_trajectories : int
        number of trajectories to use to compute the expectation values
    """

    def __init__(
        self,
        env: SimpleEnvironment,
        demonstrator_name: str,
        T: int = 45,
        n_trajectories: int = 1,
    ):
        super().__init__(env, demonstrator_name, T, n_trajectories)

        self.policy = TabularPolicy(self._define_policy())

        self.mu_demonstrator = self.get_mu_using_reward_features()

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
    n_trajectories : int
        number of trajectories to use to compute the expectation values
    """

    def __init__(
        self,
        env: MinigridEnvironment,
        demonstrator_name: str,
        T: int = 45,
        n_trajectories: int = 1,
    ):
        super().__init__(env, demonstrator_name, T, n_trajectories)

        self.policy = TabularPolicy(self._define_policy())
        self.trajectories = [
            self.solver.generate_episode(self.env, self.policy, self.T)[0]
        ]
        self.mu_demonstrator = self.get_mu_using_reward_features()

    def _define_policy(self):
        pi_s = np.zeros((self.T, self.env.n_states, self.env.n_actions))

        lava_states = self.env.env.forbidden_states

        if lava_states[0][0] == lava_states[1][0]:
            # x coordinate of the lava are the same -> vertical line
            lava_y = list(map(itemgetter(1), lava_states))

            for i in range(1, self.env.height - 1):
                if i not in lava_y:
                    hole_y = i

            if hole_y != 1:
                state = [1, 1, 0]
                state_index = self.env.env.to_state_index(*state)
                # orient downward
                pi_s[0, state_index, 1] = 1.0

                for y in range(1, hole_y):
                    state = [1, y, 1]
                    state_index = self.env.env.to_state_index(*state)
                    # move forward in these states
                    pi_s[0, state_index, 2] = 1.0

                state = [1, hole_y, 1]
                state_index = self.env.env.to_state_index(*state)
                # orient to the right
                pi_s[0, state_index, 0] = 1.0

            for x in range(1, self.env.width - 2):
                state = [x, hole_y, 0]
                state_index = self.env.env.to_state_index(*state)
                # move forward in these states
                pi_s[0, state_index, 2] = 1.0

            if hole_y != self.env.height - 2:
                state = [self.env.width - 2, hole_y, 0]
                state_index = self.env.env.to_state_index(*state)
                # orient downward
                pi_s[0, state_index, 1] = 1.0

                for y in range(hole_y, self.env.height - 2):
                    state = [self.env.width - 2, y, 1]
                    state_index = self.env.env.to_state_index(*state)
                    # move forward in these states
                    pi_s[0, state_index, 2] = 1.0

        else:
            # y coordinate of the lava are the same -> horizontal line
            lava_x = list(map(itemgetter(0), lava_states))

            for i in range(1, self.env.width - 1):
                if i not in lava_x:
                    hole_x = i

            if hole_x != 1:
                for x in range(1, hole_x):
                    state = [x, 1, 0]
                    state_index = self.env.env.to_state_index(*state)
                    # move forward in these states
                    pi_s[0, state_index, 2] = 1.0

            state = [hole_x, 1, 0]
            state_index = self.env.env.to_state_index(*state)
            # orient downward
            pi_s[0, state_index, 1] = 1.0

            for y in range(1, self.env.height - 2):
                state = [hole_x, y, 1]
                state_index = self.env.env.to_state_index(*state)
                # move forward in these states
                pi_s[0, state_index, 2] = 1.0

            state = [hole_x, self.env.height - 2, 1]
            state_index = self.env.env.to_state_index(*state)

            if hole_x != self.env.width - 2:
                # orient to the right
                pi_s[0, state_index, 0] = 1.0

                for x in range(hole_x, self.env.width - 2):
                    state = [x, self.env.height - 2, 0]
                    state_index = self.env.env.to_state_index(*state)
                    # move forward in these states
                    pi_s[0, state_index, 2] = 1.0

        for t in range(1, self.T):
            # time independent policy
            pi_s[t] = pi_s[0]

        return pi_s
