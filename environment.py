import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy import sparse
from itertools import product
import os

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)


class Environment:
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

        # initialise MDP parameters
        self.actions = {"up": 0, "left": 1, "down": 2, "right": 3}
        self.actions_names = ["up", "left", "down", "right"]
        self.n_actions = len(self.actions)
        self.grid_x = 9
        self.grid_y = 5
        self.gamma = env_args["gamma"]

        self.theta_e = env_args["theta_e"]
        self.n_features = len(
            env_args["theta_e"]
        )  # (=3) two for the objects, and whether terminal state

        self.n_states = self.grid_x * self.grid_y + 1

        self.terminal_states = self.compute_terminal_states()

        self.InitD, self.init_state_neighbourhood = self.get_initial_distribution()
        self.state_object_array = self.place_objects_on_the_grid()
        self.feature_matrix = self.get_state_feature_matrix()
        self.n_features = self.feature_matrix.shape[1]

        self.reward = self.get_reward_for_given_theta(self.theta_e)
        self.T = self.get_transition_matrix()
        self.T_sparse_list = self.get_transition_sparse_list()

    def get_reward_for_given_theta(self, theta_e):
        """
        Parameters
        ----------
        theta_e : ndarary

        Returns
        -------
        reward : ndarray
        """
        reward = self.feature_matrix.dot(theta_e)
        return np.array(reward)

    def get_variance_for_given_theta(self, theta_v):
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

    def get_transition_sparse_list(self):
        """
        computes the sparse transition list for the differnt actions

        Returns
        -------
        T_sparse_list : list[sparse.csr_matrix]
        """
        T_sparse_list = []
        T_sparse_list.append(sparse.csr_matrix(self.T[:, :, 0]))  # T_0
        T_sparse_list.append(sparse.csr_matrix(self.T[:, :, 1]))  # T_1
        T_sparse_list.append(sparse.csr_matrix(self.T[:, :, 2]))  # T_2
        T_sparse_list.append(sparse.csr_matrix(self.T[:, :, 3]))  # T_3
        return T_sparse_list

    def get_transition_matrix(self):
        """
        Computes the transition matrix for this environment
        Returns
        -------
        P : ndarray
            transition matrix
        """

        P = np.zeros((self.n_states, self.n_states, self.n_actions))

        for s in range(self.n_states):
            if s in self.terminal_states:
                P[s, self.n_states - 1, :] = 1.0
                continue

            if s == self.n_states - 1:
                P[s, s, :] = 1.0

            curr_state = s
            possible_actions = self.get_possible_actions_within_grid(s)
            next_states = self.get_next_states(curr_state, possible_actions)
            for a in range(self.n_actions):
                if a in possible_actions:
                    n_s = int(next_states[np.where(possible_actions == a)][0])
                    P[s, n_s, a] = 1.0

        return P

    def get_possible_actions_within_grid(self, state: int):
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

    def get_next_states(self, state: int, possible_actions):
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
        state_x, state_y = state // self.grid_y, state % self.grid_y
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
                if state_x < self.grid_x - 1:
                    n_state_x = n_state_x + 1
            if a == 3:
                if state_y < self.grid_y - 1:
                    n_state_y = n_state_y + 1
            next_state.append(n_state_x * self.grid_y + n_state_y)

        next_state = np.array(next_state, dtype=int)
        return next_state

    def get_initial_distribution(self):
        """
        computes initial state distribution and the 1x1 neighborhood of the starting state

        Returns
        -------
        initial_dist : ndarray
        init_state_neighbourhood : list[int]
        """
        init_state_neighbourhood = []
        initial_dist = np.zeros(self.n_states)

        self.init_state = self.point_to_int(4, 0)

        initial_dist[self.init_state] = 1.0

        x, y = self.int_to_point(self.init_state)
        # compute states which should be in one 1x1 neighborhood of init state

        for dx, dy in product(
            range(-1, 2), range(-1, 2)
        ):  # get 1x1 neighborhood init state
            if 0 <= x + dx < self.grid_x and 0 <= y + dy < self.grid_y:
                neighbour_x = x + dx
                neighbour_y = y + dy
                neighbour_state = self.point_to_int(neighbour_x, neighbour_y)
                init_state_neighbourhood.append(neighbour_state)
        return initial_dist, init_state_neighbourhood

    def compute_terminal_states(self):
        """
        Stores the indices of the target states

        Returns
        -------
        terminal_states : list[int]
            list of the target states
        """

        terminal_states = []
        terminal_states.append(self.point_to_int(0, 0))
        terminal_states.append(self.point_to_int(self.grid_x - 1, 0))
        terminal_states.append(self.point_to_int(4, self.grid_y - 1))

        return terminal_states

    def place_objects_on_the_grid(self):
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

    def int_to_point(self, i: int):
        """
        Returns
        -------
        tuple[int,int] :  representing the coordinate for the given state
        """
        return (i // self.grid_y, i % self.grid_y)

    def point_to_int(self, x: int, y: int):
        """
        Returns
        -------
        int : index of the state within the enumeration given its coordinates
        """
        return x * self.grid_y + y

    def get_state_feature_vector_full(self, state: int):
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

    def get_state_feature_matrix(self):
        """
        Returns
        -------
        feature_matrix : ndarray
            representing full feature matrix
        """
        feature_matrix = np.zeros((self.n_states, self.n_features))
        for i in range(self.n_states):
            feature_matrix[i, :] = self.get_state_feature_vector_full(i)
        return feature_matrix

    def get_nxn_neighborhood(self, neighborhood: int):
        """
        Parameters
        ----------
        neighborhood : int
            size of the neighborhood

        Returns
        -------
        only_outer_neighborhood : list[tuple[int,int]]
        """
        outer = list(
            product(
                range(-neighborhood, neighborhood + 1),
                range(-neighborhood, neighborhood + 1),
            )
        )
        inner = list(
            product(
                range(-neighborhood + 1, neighborhood),
                range(-neighborhood + 1, neighborhood),
            )
        )

        only_outer_neighborhood = [x for x in outer if x not in inner]
        return only_outer_neighborhood

    def draw(
        self,
        V,
        pi,
        reward,
        show: bool = False,
        strname: str = "",
        fignum: int = 0,
        store: bool = False,
    ):
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
        strname : str
            plot title
        fignum : int
            figure identifier (default = 0)
        store : bool
            whether or not plots should be store (default = False)
        """
        f = fignum
        plt.figure(f)

        states = self.n_states - 1

        reward = reward[:-1]
        V = V[:, :-1]
        pi = pi[:-1]

        reshaped_reward = copy.deepcopy(reward.reshape((self.grid_x, self.grid_y)))
        reshaped_reward = np.flip(reshaped_reward, 0)
        plt.pcolor(reshaped_reward)
        plt.colorbar()
        plt.title(strname + ": reward function")
        if show:
            plt.show()
        if store:
            plt.savefig(f"plots\{strname}_reward.jpg", format="jpg")

        if V is not None:

            x = np.linspace(0, self.grid_y - 1, self.grid_y) + 0.5
            y = np.linspace(self.grid_x - 1, 0, self.grid_x) + 0.5
            X, Y = np.meshgrid(x, y)
            zeros = np.zeros((self.grid_x, self.grid_y))
            f += 1
            plt.figure(f)
            V_plot = V[0, :]
            reshaped_Value = copy.deepcopy(V_plot.reshape((self.grid_x, self.grid_y)))
            reshaped_Value = np.flip(reshaped_Value, 0)
            plt.pcolor(reshaped_Value, vmin=-10)
            plt.colorbar()
            if pi is not None:
                current_states = [self.init_state]
                visited = []
                for t in range(pi.shape[0]):
                    for state in current_states:
                        coord = self.int_to_point(state)
                        plt.text(coord[1], coord[0], t, color="black")
                    visited += current_states
                    for a in range(self.n_actions):
                        pi_ = np.zeros(states)
                        for s in current_states:
                            if np.max(pi[t, s, :]) > 0:
                                pi_[s] = 0.45 * pi[t, s, a] / np.max(pi[t, s, :])

                        pi_ = pi_.reshape(self.grid_x, self.grid_y)
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
                            for x in self.get_next_states(
                                n, self.get_possible_actions_within_grid(n)
                            )
                            if x not in visited
                        )
                    )

            plt.title(strname + ": optimal values and policy")
            if show:
                plt.show()
            if store:
                plt.savefig(
                    os.path.join("plots", f"{strname}_policy.jpg"), format="jpg"
                )

            plt.close()


if __name__ == "__main__":

    config_env = {
        "theta_e": [1.0, 1.0, -2.0],
        "gamma": 1.0,
    }

    env = Environment(config_env)

    env.draw(None, pi=None, reward=env.reward, show=True, strname="test")
