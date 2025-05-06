import numpy as np
import matplotlib.pyplot as plt
from policy import Policy
import copy
import os
from environments.environment import Environment

class SimpleEnvironment(Environment):
    """
    implements a grid world with different objects that an agent will have to collect

    Parameters
    ----------
    env_args: dict[Any]
        environment definition parameters consisting of:
        - gamma : MDP discounting factor
        - theta_e : the real reward parameters
    """

    def __init__(self, env_args: dict):
        super().__init__(env_args)

        # initialise MDP parameters
        self.actions = {"up": 0, "left": 1, "down": 2, "right": 3}
        self.actions_names = ["up", "left", "down", "right"]
        self.n_actions = len(self.actions)
        self.width = 9
        self.height = 5
        self.theta_reward = env_args["theta"]

        self.n_features = len(
            env_args["theta"]
        )  # (=3) two for the objects, and whether terminal state

        self.n_states = self.width * self.height + 1

        self.InitD = self._get_initial_distribution()

        self.state_object_array = self.__place_objects_on_the_grid()

        self.T_matrix, self.terminat_states = self._compute_transition_matrix()
        self.T_sparse_list = self._compute_transition_sparse_list()
        self.feature_matrix = self._compute_state_feature_matrix()

        self.reward = self.get_reward_for_given_theta(self.theta_reward)
        
        self.agent_position = int(np.random.choice(np.arange(self.n_states), p=self.InitD))


    def _compute_state_feature_matrix(self) -> np.ndarray:
        """
        Returns
        -------
        feature_matrix : ndarray
            representing full feature matrix
        """
        feature_matrix = np.zeros((self.n_states, self.n_features))
        for i in range(self.n_states):
            feature_matrix[i, :] = self.__get_state_feature_vector_full(i)
        return feature_matrix
    
    def _compute_transition_matrix(self) -> tuple[np.ndarray, list[int]]:
        """
        Computes the transition matrix for this environment
        Returns
        -------
        P : ndarray
            transition matrix
        terminal_states : list
            terminal states of the environment
        """

        P = np.zeros((self.n_states, self.n_states, self.n_actions))
        self.terminal_states = self.__compute_terminal_states()

        for s in range(self.n_states):
            if s in self.terminal_states:
                P[s, self.n_states-1, :] = 1.0
                continue

            if s == self.n_states - 1:
                P[s,s,:] = 1.0

            curr_state = s
            possible_actions = self.__get_possible_actions_within_grid(s)
            next_states = self.__get_next_states(curr_state, possible_actions)
            for a in range(self.n_actions):
                if a in possible_actions:
                    n_s = int(next_states[np.where(possible_actions == a)][0])
                    P[s, n_s, a] = 1.0

        return P, self.terminal_states
    
    def _get_initial_distribution(self) -> np.ndarray:
        """
        computes initial state distribution

        Returns
        -------
        initial_dist : ndarray
        """
        initial_dist = np.zeros(self.n_states)

        self.init_state = self.point_to_int(4, 0)
        initial_dist[self.init_state] = 1.0

        
        return initial_dist
    
    def reset(self) -> any:
        """
        Reset wrapper to generalize environment access over different environments

        Returns
        -------
        Initial state description
        """
        self.agent_position = int(np.random.choice(np.arange(self.n_states), p=self.InitD))
        return self.agent_position
    
    def step(self, action:int) -> tuple[any, float, bool, bool]:
        """
        Step wrapper to generalize environment access over different environments

        Parameters
        ----------
        action : int
            action to take

        Returns
        -------
        new_state 
            current state description
        reward : float
            reward for taken action
        terminated : bool
            if episode is terminated
        truncated : bool
            if episode was truncated
        """
        next_state_prob = self.T_matrix[self.agent_position, :, action]
        new_state = int(np.random.choice(np.arange(self.n_states), p=next_state_prob))
        self.agent_position = new_state

        return new_state, self.reward[new_state], new_state in self.terminal_states, new_state in self.terminal_states

    def render(
        self,
        policy:Policy,
        T:int = 20,
        store: bool = False,  
        reward:np.ndarray = None,
        V:np.ndarray = None,
        show: bool = False,
        strname: str = "",
        fignum: int = 0,
        **kwargs  
    ) -> None:
        """
        draws a given policy and reward in the gridworld

        Parameters
        ----------
        V : ndarray
            value function
        reward : ndarray
            reward for the different states
        show : bool
            wheter or not to show the plot
        store : bool
            whether or not plots should be store
        strname : str
            plot title
        fignum : int
            figure identifier
        
        """
        f = fignum
        plt.figure(f)

        states = self.n_states - 1

        reward = reward[:-1]
        V = V[:,:-1]
        pi = policy.pi[:-1]


        reshaped_reward = copy.deepcopy(reward.reshape((self.width, self.height)))
        reshaped_reward = np.flip(reshaped_reward, 0)
        plt.pcolor(reshaped_reward)
        plt.colorbar()
        plt.title(strname + ": reward function")
        if show:
            plt.show()
        if store:
            plt.savefig(f"plots\simple_environment\{strname}_reward.jpg", format="jpg")

        if V is not None:

            x = np.linspace(0, self.height - 1, self.height) + 0.5
            y = np.linspace(self.width - 1, 0, self.width) + 0.5
            X, Y = np.meshgrid(x, y)
            zeros = np.zeros((self.width, self.height))
            f += 1
            plt.figure(f)
            V_plot = V[0, :]
            reshaped_Value = copy.deepcopy(V_plot.reshape((self.width, self.height)))
            reshaped_Value = np.flip(reshaped_Value, 0)
            plt.pcolor(reshaped_Value, vmin=-10)
            plt.colorbar()
            if pi is not None:
                current_states = [self.init_state]
                visited = []
                for t in range(pi.shape[0]):
                    for state in current_states:
                        coord = self.__int_to_point(state)
                        plt.text(coord[1], coord[0], t,  color='black')
                    visited += current_states
                    for a in range(self.n_actions):
                        pi_ = np.zeros(states)
                        for s in current_states:
                            if np.max(pi[t, s, :]) > 0:
                                pi_[s] = 0.45 * pi[t, s, a] / np.max(pi[t, s, :])

                        pi_ = pi_.reshape(self.width, self.height)
                        if a == 2:
                            plt.quiver(X, Y, zeros, -pi_, scale=1, units="xy")
                        elif a == 1:
                            plt.quiver(X, Y, -pi_, zeros, scale=1, units="xy")
                        elif a == 0:
                            plt.quiver(X, Y, zeros, pi_, scale=1, units="xy")
                        elif a == 3:
                            plt.quiver(X, Y, pi_, zeros, scale=1, units="xy")
                    current_states = list(
                        set(
                            x
                            for n in current_states
                            for x in self.__get_next_states(
                                n, self.__get_possible_actions_within_grid(n)
                            )
                            if x not in visited
                        )
                    )

            plt.title(strname + ": optimal values and policy")
            if show:
                plt.show()
            if store:
                plt.savefig(os.path.join("plots", f"{strname}_policy.jpg"), format="jpg")

            plt.close()

    def __get_possible_actions_within_grid(self, state: int) -> np.ndarray:
        """
        computes possible actions from a given state

        Parameters
        ----------
        state : int
            current state

        Returns
        -------
        posible_actions : ndarray
            indices of possible actions
        """

        possible_actions = []

        if state in self.terminal_states:
            return np.array(possible_actions, dtype=int)

        possible_actions = np.array(list(range(4)), dtype=int)
        return possible_actions

    def __get_next_states(self, state: int, possible_actions: np.ndarray) -> np.ndarray:
        """
        computes the next possible states

        Parameters
        ----------
        state : int
            current state
        possible_actions : ndarray
            actions possible from the given state

        Returns
        -------
        next_state : ndarray
            indices of possible next states
        """
        next_state = []

        # {"up": 0, "left": 1, "down": 2, "right": 3}
        state_x, state_y = state // self.height, state % self.height
        for a in possible_actions:
            if state == self.n_states - 1:
                next_state.append(state)
                continue

            n_state_x = state_x
            n_state_y = state_y
            if a == 0:
                if state_x > 0:
                    n_state_x = n_state_x - 1
            if a == 1:
                if state_y > 0:
                    n_state_y = n_state_y - 1
            if a == 2:
                if state_x < self.width - 1:
                    n_state_x = n_state_x + 1
            if a == 3:
                if state_y < self.height - 1:
                    n_state_y = n_state_y + 1
            next_state.append(n_state_x * self.height + n_state_y)

        next_state = np.array(next_state, dtype=int)
        return next_state

    def __compute_terminal_states(self) -> list[int]:
        """
        Stores the indices of the target states

        Returns
        -------
        terminal_states : list[int]
            list of the target states
        """

        terminal_states = []
        terminal_states.append(self.point_to_int(0, 0))
        terminal_states.append(self.point_to_int(self.width - 1, 0))
        terminal_states.append(self.point_to_int(4, self.height - 1))

        return terminal_states

    def __place_objects_on_the_grid(self) -> np.ndarray:
        """
        Returns the array indicating where te different objects are placed
        -------
        state_object_array : ndarray
        """

        state_object_array = np.zeros((self.n_states, 3))

        # place the objects

        # object 0 = triangle
        # object 1 = diamond
        # object 2 = square
        state_object_array[self.point_to_int(2, 0), 0] = 1

        state_object_array[self.point_to_int(4, 2), 1] = 1
        state_object_array[self.point_to_int(6, 0), 2] = 1

        return state_object_array

    def __int_to_point(self, i: int) -> tuple[int, int]:
        """
        Returns
        -------
        tuple[int,int] :  representing the coordinate for the given state
        """
        return (i // self.height, i % self.height)

    def point_to_int(self, x: int, y: int) -> int:
        """
        Returns
        -------
        int : index of the state within the enumeration given its coordinates
        """
        return x * self.height + y

    def __get_state_feature_vector_full(self, state: int) -> np.ndarray:
        """
        Returns
        -------
        feature_vector : ndarray
            represents the features of the given state
        """
        feature_vector = np.zeros(self.n_features)

        if state == self.n_states - 1:
            return feature_vector

        # get feature for objects (is present on this state)
        if self.state_object_array[state, 0] != 0:
            feature_vector[0] = 2

        if self.state_object_array[state, 1] != 0:
            feature_vector[0] = 1
            feature_vector[1] = 1

        if self.state_object_array[state, 2] != 0:
            feature_vector[1] = 2

        if state in self.terminal_states:
            feature_vector[0] = 1
            feature_vector[1] = 1
            feature_vector[2] = 1  # features that state is the target state
        else:
            feature_vector[-1] = 1  # cost of taking any other step

        return feature_vector
