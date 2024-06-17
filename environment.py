import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy import sparse
import matplotlib.pyplot as plt
from itertools import product

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
        - object_rewards : the rewards for the different kinds of objects
    """

    def __init__(self, env_args: dict):

        # initialise MDP parameters
        self.actions = {"up": 0, "left": 1, "down": 2, "right": 3}
        self.actions_names = ["up", "left", "down", "right"]
        self.n_actions = len(self.actions)
        self.grid_size_full = 6
        self.gamma = env_args["gamma"]

        self.n_features_reward = len(env_args["theta_e"])
        self.theta_e = env_args["theta_e"]
        self.theta_v = env_args["theta_v"]
        self.n_features_full = len(
            env_args["theta_e"]
        )  # (=2) one for each object, and whether terminal state

        self.n_states = self.grid_size_full * self.grid_size_full

        self.object_rewards = env_args["object_rewards"]

        self.InitD, self.init_state_neighbourhood = self.get_initial_distribution()
        self.state_object_array = self.place_objects_on_the_grid()
        self.feature_matrix_full = self.get_state_feature_matrix_full()

        self.feature_matrix = self.get_state_feature_matrix()
        self.n_features = self.feature_matrix.shape[1]

        self.reward = self.get_reward_for_given_theta(self.theta_e)
        self.variance = self.get_variance_for_given_theta(np.array(self.theta_v))
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
            theta_v.dot(self.feature_matrix[i, :]).dot(self.feature_matrix[i, :])
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
        # Contructs the Transition Matrix
        # Explicitly put n_states = self.grid_size_full_x * self.grid_size_full_y
        P = np.zeros((self.n_states, self.n_states, self.n_actions))
        # last state is the goal state -> no possible actions from there
        for s in range(self.n_states - 1):
            curr_state = s
            possible_actions = self.get_possible_actions_within_grid(s)
            next_states = self.get_next_states(curr_state, possible_actions)
            for a in range(self.n_actions - 1):
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

        if state == self.n_states - 1:
            return np.array(possible_actions, dtype=int)

        state_x, state_y = state // self.grid_size_full, state % self.grid_size_full
        if state_x > 0:
            possible_actions.append(self.actions["up"])
        if state_x < self.grid_size_full - 1:
            possible_actions.append(self.actions["down"])
        if state_y > 0:
            possible_actions.append(self.actions["left"])
        if state_y < self.grid_size_full - 1:
            possible_actions.append(self.actions["right"])

        possible_actions = np.array(possible_actions, dtype=int)
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

        state_x, state_y = state // self.grid_size_full, state % self.grid_size_full
        for a in possible_actions:
            if a == 0:
                next_state.append((state_x - 1) * self.grid_size_full + state_y)
            if a == 1:
                next_state.append(state_x * self.grid_size_full + state_y - 1)
            if a == 2:
                next_state.append((state_x + 1) * self.grid_size_full + state_y)
            if a == 3:
                next_state.append(state_x * self.grid_size_full + state_y + 1)

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

        init_state = 0

        initial_dist[init_state] = 1.0

        x, y = self.int_to_point(init_state)
        # compute states which should be in one 1x1 neighborhood of init state

        for dx, dy in product(
            range(-1, 2), range(-1, 2)
        ):  # get 1x1 neighborhood init state
            if 0 <= x + dx < self.grid_size_full and 0 <= y + dy < self.grid_size_full:
                neighbour_x = x + dx
                neighbour_y = y + dy
                neighbour_state = self.point_to_int(neighbour_x, neighbour_y)
                init_state_neighbourhood.append(neighbour_state)
        return initial_dist, init_state_neighbourhood

    def place_objects_on_the_grid(self):
        """
        Returns the array indicating where te different objects are placed
        -------
        state_object_array : ndarray
        """
        states = self.grid_size_full * self.grid_size_full

        state_object_array = np.zeros((states, 1))

        for i in range(1, self.grid_size_full - 1):
            # object 0 on the entire upper boundary
            state_object_array[self.point_to_int(0, i)] += self.object_rewards[0]

        for i in range(self.grid_size_full):
            # object 0 on the entire right boundary
            state_object_array[
                self.point_to_int(i, self.grid_size_full - 1)
            ] += self.object_rewards[0]

        # place the rest of the objects
        state_object_array[self.point_to_int(1, 0)] += self.object_rewards[0]

        state_object_array[self.point_to_int(1, 1)] += self.object_rewards[1]
        state_object_array[self.point_to_int(3, 3)] += self.object_rewards[1]

        state_object_array[self.point_to_int(4, 0)] += self.object_rewards[2]
        state_object_array[self.point_to_int(5, 2)] += self.object_rewards[2]
        state_object_array[self.point_to_int(5, 3)] += self.object_rewards[2]
        state_object_array[self.point_to_int(4, 4)] += self.object_rewards[2]

        return state_object_array

    def int_to_point(self, i: int):
        """
        Returns
        -------
        tuple[int,int] :  representing the coordinate for the given state
        """
        return (i // self.grid_size_full, i % self.grid_size_full)

    def point_to_int(self, x: int, y: int):
        """
        Returns
        -------
        int : index of the state within the enumeration given its coordinates
        """
        return x * self.grid_size_full + y

    def get_state_feature_vector_full(self, state: int):
        """
        Returns
        -------
        feature_vector : ndarray
            represents the features of the given state
        """
        feature_vector = np.zeros(self.n_features_full)

        # get feature for objects (is present on this state)
        if self.state_object_array[state] != 0:
            feature_vector[0] = self.state_object_array[state]

        if state == self.n_states - 1:
            feature_vector[-1] = 1  # feature that state is the target state
        else:
            feature_vector[-1] = -1 # cost of taking any other step

        return feature_vector

    def get_state_feature_matrix_full(self):
        """
        Returns
        -------
        feature_matrix : ndarray
            representing full feature matrix
        """
        states = self.grid_size_full * self.grid_size_full
        feature_matrix = np.zeros((self.n_states, self.n_features_full))
        for i in range(states):
            feature_matrix[i, :] = self.get_state_feature_vector_full(i)
        return feature_matrix

    def get_state_feature_matrix(self):
        """
        Returns
        -------
        new_feature_matrix : ndarray
            representing the feature matrix restricted to reward features (for our example same as whole matrix)
        """
        index_of_reward_features = range(0, self.n_features_reward)
        new_feature_matrix = self.feature_matrix_full[:, index_of_reward_features]
        return new_feature_matrix

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
        self, V, pi, reward, show: bool = False, strname: str = "", fignum: int = 0, store: bool = False
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

        states = self.n_states

        reshaped_reward = copy.deepcopy(
            reward.reshape((self.grid_size_full, self.grid_size_full))
        )
        reshaped_reward = np.flip(reshaped_reward, 0)
        plt.pcolor(reshaped_reward)
        plt.colorbar()
        plt.title(strname + ": reward function")
        if store:
            plt.savefig(f"plots\{strname}_reward.jpg", format="jpg")
        
        f += 1
        if V is not None:
            plt.figure(f)
            V = self.get_V_for_plotting(V)
            reshaped_Value = copy.deepcopy(
                V.reshape((self.grid_size_full, self.grid_size_full))
            )
            reshaped_Value = np.flip(reshaped_Value, 0)
            plt.pcolor(reshaped_Value)
            plt.colorbar()
            x = np.linspace(0, self.grid_size_full - 1, self.grid_size_full) + 0.5
            y = np.linspace(self.grid_size_full - 1, 0, self.grid_size_full) + 0.5
            X, Y = np.meshgrid(x, y)
            zeros = np.zeros((self.grid_size_full, self.grid_size_full))
            if pi is not None:
                current_states = [0]
                visited = []
                for t in range(pi.shape[0]):
                    visited += current_states
                    for a in range(self.n_actions):
                        pi_ = np.zeros(states)
                        for s in current_states:
                            if np.max(pi[t, s, :]) > 0:
                                pi_[s] = 0.45 * pi[t, s, a] / np.max(pi[t, s, :])

                        pi_ = pi_.reshape(self.grid_size_full, self.grid_size_full)
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
            if store:
                plt.savefig(f"plots\{strname}_policy.jpg", format="jpg")
        if show:
            plt.show()

    def get_V_for_plotting(self, V):
        """
        Extracts necessary values of the time dependent value function for plotting

        Parameters
        ----------
        V : ndarray
            time dependent value function

        Returns
        -------
        ndarray of one dimension less containing some of the values from the value function
        """
        result = np.zeros(self.n_states)

        current_states = [0]
        visited = []
        for t in range(V.shape[0]):
            visited += current_states
            for s in current_states:
                result[s] = V[t, s]
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
        return result


if __name__ == "__main__":

    config_env2 = {
        "theta_e": [1.0, -1.0],
        "theta_v": [[0.0, 0.0], [0.0, 0.0]],
        "gamma": 1.0,
        "object_rewards": [0.5, 0.25, 1.75],
    }

    env = Environment(config_env2)

    env.draw(None, pi=None, reward=env.reward, show=True, strname="test")
