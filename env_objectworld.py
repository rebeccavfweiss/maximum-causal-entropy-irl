import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import copy
from scipy import sparse
import matplotlib.pyplot as plt
import MDPSolver
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
        - roadlength: Length of Road
        - terminalState: Do we have a self-absorbing state
        - gamma: MDP discounting factor
        - n_template: Number of templates we want to create
        - template_list = [0,1,0,2]  Specifications of what we want
        - terminal_gamma: Parameter used for T matrix for self-absorbing states
        - randomMoveProb: probability mass on random move
        - ...
        
    rng: 
        random number generator
    """

    def __init__(self, env_args, rng=None):

        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng # random number generator

        # initialise MDP parameters
        self.actions = {"up": 0, "left": 1, "down": 2, "right": 3, "stay": 4}
        self.actions_names = ["up", "left", "down", "right", "stay"]
        self.n_actions = len(self.actions)
        self.grid_size_full = env_args["gridsizefull"]
        self.gamma = env_args["gamma"]
        self.n_objects = 3
        self.n_objects_each = 2

        self.n_features_reward = len(env_args["theta_e"])
        self.theta_e = env_args["theta_e"]
        self.theta_v = env_args["theta_v"]
        self.terminal_state = env_args["terminalState"]
        self.randomMoveProb = env_args["randomMoveProb"]
        self.terminal_gamma = env_args["terminal_gamma"]
        self.n_features_full = 9
        self.n_templates = 1

        if self.terminal_state == 1:
            self.n_states = self.grid_size_full * self.grid_size_full + 1
            self.Hmax = self.Hmax = 1/(1-self.terminal_gamma)
        else:
            self.n_states = self.grid_size_full * self.grid_size_full
            self.Hmax = 1 / (1 - self.gamma)

        self.InitD, self.init_state_neighbourhood = self.get_initial_distribution()
        self.state_object_array, self.states_for_object = self.place_objects_on_the_grid()
        self.feature_matrix_full = self.get_state_feature_matrix_full()

        self.feature_matrix = self.get_state_feature_matrix()
        self.n_features = self.feature_matrix.shape[1]
        
        self.reward = self.get_reward_for_given_theta(self.theta_e)
        self.variance = self.get_variance_for_given_theta(np.array(self.theta_v))
        self.T = self.get_transition_matrix(self.randomMoveProb)
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
        variance = [self.gamma**i *  theta_v.dot(self.feature_matrix[i,:]).dot(self.feature_matrix[i,:]) for i in range(self.feature_matrix.shape[0])]

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
        T_sparse_list.append(sparse.csr_matrix(self.T[:, :, 4]))  # T_4
        return T_sparse_list

    def get_transition_matrix(self, randomMoveProb:float):
        """
        Parameters
        ----------
        randomMoveProb : float
            probability for making a random move

        Returns
        -------
        P : ndarray
            transition matrix
        """
        # Contructs the Transition Matrix
        # Explicitly put n_states = self.grid_size_full_x * self.grid_size_full_y
        states = self.grid_size_full * self.grid_size_full
        P = np.zeros((states, states, self.n_actions))
        for s in range(states):
            curr_state = s
            possible_actions = self.get_possible_actions_within_grid(s)
            next_states = self.get_next_states(curr_state, possible_actions)
            for a in range(self.n_actions - 1):
                if a in possible_actions:
                    n_s = int(next_states[np.where(possible_actions == a)][0])
                    P[s, n_s, a] = 1.0 - randomMoveProb
                    next_states_copy = np.setdiff1d(next_states, n_s)
                    for i in next_states_copy:
                        P[s, i, a] = randomMoveProb / next_states_copy.size
                if a not in possible_actions:
                    for i in next_states:
                        P[s, i, a] = randomMoveProb / next_states.size
                    P[s, s, a] = 1 - randomMoveProb

        # stay action
        for s in range(states):
            P[s, s, self.n_actions - 1] = 1

        if self.terminal_state == 1:
            # Now, if the MDP, has a terminal state, 4 things have to be taken care of
            # Initialise a newP with different dimensions
            P_changed = np.zeros((self.n_states, self.n_states, self.n_actions))
            # Copy the old P as it is in this new P
            P_changed[0:P.shape[0], 0:P.shape[1]] = P
            # Now, I will set the probability of going from any object_state with all actions to terminal state as 1-\gamma
            P_changed[self.states_for_object, self.n_states - 1, :] = 1 - self.terminal_gamma
            # For the terminal state, all actions lead to the terminal state itself
            P_changed[self.n_states - 1, self.n_states - 1, :] = 1

            P_changed[self.states_for_object, :-1, :] *= self.terminal_gamma
            return P_changed
        else:
            return P

    def get_possible_actions_within_grid(self, state:int):
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
        state_x, state_y = state // self.grid_size_full, state % self.grid_size_full
        if ((state_x > 0)): possible_actions.append(self.actions["up"])
        if ((state_x < self.grid_size_full - 1)): possible_actions.append(self.actions["down"])
        if ((state_y > 0)): possible_actions.append(self.actions["left"])
        if ((state_y < self.grid_size_full - 1)): possible_actions.append(self.actions["right"])

        possible_actions = np.array(possible_actions, dtype=int)
        return possible_actions

    def get_next_states(self, state:int, possible_actions):
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
            if a == 0: next_state.append((state_x - 1) * self.grid_size_full + state_y)
            if a == 1: next_state.append(state_x * self.grid_size_full + state_y - 1)
            if a == 2: next_state.append((state_x + 1) * self.grid_size_full + state_y)
            if a == 3: next_state.append(state_x * self.grid_size_full + state_y + 1)

        next_state = np.array(next_state, dtype=int)
        return next_state

    def get_initial_distribution(self):
        """
        computes initial state distribution 
        
        Returns
        -------
        initial_dist : ndarray
        init_state_neighbourhood : list[int]
        """
        init_state_neighbourhood = []
        states = self.grid_size_full * self.grid_size_full
        initial_dist = np.zeros(self.n_states)
        if self.grid_size_full % 2 == 1:
            init_state = states//2
            initial_dist[init_state] = 1

        else:
            init_states_array = []
            central_state_1d = states // 2
            central_state_2d_1 = central_state_1d - (self.grid_size_full - 1)//2 - 1
            init_states_array.append(central_state_2d_1)
            central_state_2d_2 = central_state_1d - (self.grid_size_full - 1) // 2 - 2
            init_states_array.append(central_state_2d_2)
            central_state_2d_3 = central_state_1d + (self.grid_size_full - 1) // 2 + 0
            init_states_array.append(central_state_2d_3)
            central_state_2d_4 = central_state_1d + (self.grid_size_full - 1) // 2 + 1
            init_states_array.append(central_state_2d_4)

            init_state = self.rng.choice(init_states_array, p=np.ones(len(init_states_array))/len(init_states_array))
            initial_dist[init_state] = 1

        x, y = self.int_to_point(init_state)
        # compute states which should be in one 1x1 neighborhood of init state

        for (dx, dy) in product(range(-1, 2), range(-1, 2)):  # get 1x1 neighborhood init state
            if 0 <= x + dx < self.grid_size_full and 0 <= y + dy < self.grid_size_full:
                neighbour_x = x + dx
                neighbour_y = y + dy
                neighbour_state = self.point_to_int(neighbour_x, neighbour_y)
                init_state_neighbourhood.append(neighbour_state)
        return initial_dist, init_state_neighbourhood

    def place_objects_on_the_grid(self):
        """
        Returns
        -------
        state_object_array : ndarray
        states_for_object : ndarray
        """
        states = self.grid_size_full * self.grid_size_full

        state_object_array = np.zeros((states, self.n_objects))
        states_to_choose_from = np.setdiff1d(range(states), self.init_state_neighbourhood)
        # choose and fix states for each objects randomly
        states_for_object = self.rng.choice(states_to_choose_from, size=self.n_objects_each * self.n_objects, replace=False)
        for i, s in enumerate(states_for_object):
            state_object_array[s, i%self.n_objects] = 1
        return state_object_array, states_for_object


    def int_to_point(self, i:int):
        """
        Convert a state int into the corresponding coordinate.
        i: State int.
        -> (x, y) int tuple.
        """
        return (i // self.grid_size_full, i % self.grid_size_full)


    def point_to_int(self, x:int, y:int):
        """
        Convert a state (x,y) into the corresponding coordinate i.
        x, y: (x, y) int tuple..
        -> state - > int
        """
        return (x * self.grid_size_full + y)

    def get_state_feature_vector_full(self, state:int):
        """
        Get the feature vector associated with a state integer.
        i: State int.
        -> Feature vector.
        """
        feature_vector = np.zeros(self.n_features_full)

        #get feature for objects (is present on this state)
        for obj in range(self.n_objects):
            if self.state_object_array[state, obj] == 1:
                feature_vector[obj] = 1

        return feature_vector

    def get_state_feature_matrix_full(self):
        """
        Returns
        -------
        feature_matrix : ndarray
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
        """
        index_of_reward_features = range(0, self.n_features_reward)
        new_feature_matrix = self.feature_matrix_full[:,index_of_reward_features]
        return new_feature_matrix


    def get_nxn_neighborhood(self, neighborhood:int):
        """
        Parameters
        ----------
        neighborhood : int
            size of the neighborhood

        Returns
        -------
        only_outer_neighborhood : list[tuple[int,int]]
        """
        outer = list(product(range(-neighborhood, neighborhood+1), range(-neighborhood, neighborhood+1)))
        inner = list(product(range(-neighborhood + 1 , neighborhood), range(-neighborhood + 1, neighborhood)))

        only_outer_neighborhood = [x for x in outer if x not in inner]
        return only_outer_neighborhood

    def draw(self, V, pi, reward, show:bool, strname:str, fignum:int = 0):
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
        """
        f = fignum
        plt.figure(f)

        if self.terminal_state == 1:
            reward = reward[:-1]
            V = V[:-1]
            pi = pi[:-1]
            states = self.n_states -1
        else:
            states = self.n_states

        reshaped_reward = copy.deepcopy(reward.reshape((self.grid_size_full, self.grid_size_full)))
        reshaped_reward = np.flip(reshaped_reward, 0)
        plt.pcolor(reshaped_reward)
        plt.title(strname + "Reward")
        plt.colorbar()
        f += 1
        if V is not None:
            plt.figure(f)
            reshaped_Value = copy.deepcopy(V.reshape((self.grid_size_full, self.grid_size_full)))
            reshaped_Value = np.flip(reshaped_Value, 0)
            plt.pcolor(reshaped_Value)
            plt.colorbar()
            x = np.linspace(0, self.grid_size_full - 1, self.grid_size_full) + 0.5
            y = np.linspace(self.grid_size_full - 1, 0, self.grid_size_full) + 0.5
            X, Y = np.meshgrid(x, y)
            zeros = np.zeros((self.grid_size_full, self.grid_size_full))
            if pi is not None:
                for a in range(self.n_actions):
                    pi_ = np.zeros(states)
                    for s in range(states):
                        if(np.max(pi[s, :]) > 0):
                            pi_[s] = 0.45*pi[s, a]/np.max(pi[s, :])

                    pi_ = (pi_.reshape(self.grid_size_full, self.grid_size_full))
                    if a == 2:
                        plt.quiver(X, Y, zeros, -pi_, scale=1, units='xy')
                    elif a == 1:
                        plt.quiver(X, Y, -pi_, zeros, scale=1, units='xy')
                    elif a == 0:
                        plt.quiver(X, Y, zeros, pi_, scale=1, units='xy')
                    elif a == 3:
                        plt.quiver(X, Y, pi_, zeros, scale=1, units='xy')

            plt.title(strname + "Opt values and policy")
        if(show == True):
            plt.show()