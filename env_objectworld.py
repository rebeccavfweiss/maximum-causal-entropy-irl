import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import copy
import math
import time
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.linalg import block_diag
import MDPSolver
from itertools import product

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)

class Environment:
    # Explanation for the Environment Definition Parameters
    # roadlength: Length of Road
    # terminalState: Do we have a self-absorbing state
    # gamma: MDP discounting factor
    # n_template: Number of templates we want to create
    # template_list = [0,1,0,2] # Specifications of what we want
    # terminal_gamma: Parameter used for T matrix for self-absorbing states
    # randomMoveProb: probability mass on random move

    def __init__(self, env_args, rng=None):

        """
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

        # convert feature matrix according to learner's preference feature
        self.feature_matrix = self.get_state_feature_matrix()
        self.n_features = self.feature_matrix.shape[1]
        
        #self.w_star = self.get_w_star()
        self.reward = self.get_reward_for_given_theta(self.theta_e)
        self.variance = self.get_variance_for_given_theta(np.array(self.theta_v))
        self.T = self.get_transition_matrix(self.randomMoveProb)
        self.T_sparse_list = self.get_transition_sparse_list()


    def get_reward_for_given_theta(self, theta_e):
        reward = self.feature_matrix.dot(theta_e)
        return np.array(reward)
    
    def get_variance_for_given_theta(self, theta_v):
        variance = [self.gamma**i *  theta_v.dot(self.feature_matrix[i,:]).dot(self.feature_matrix[i,:]) for i in range(self.feature_matrix.shape[0])]

        return np.array(variance)

    def frobenius_inner(self, a,b):
        return np.trace(np.matmul(a.T,b))

    def get_transition_sparse_list(self):
        T_sparse_list = []
        T_sparse_list.append(sparse.csr_matrix(self.T[:, :, 0]))  # T_0
        T_sparse_list.append(sparse.csr_matrix(self.T[:, :, 1]))  # T_1
        T_sparse_list.append(sparse.csr_matrix(self.T[:, :, 2]))  # T_2
        T_sparse_list.append(sparse.csr_matrix(self.T[:, :, 3]))  # T_3
        T_sparse_list.append(sparse.csr_matrix(self.T[:, :, 4]))  # T_4
        return T_sparse_list

    def get_transition_matrix(self, randomMoveProb):
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

    def get_possible_actions_within_grid(self, state):
        # Given a state, what are the possible actions from it
        possible_actions = []
        state_x, state_y = state // self.grid_size_full, state % self.grid_size_full
        if ((state_x > 0)): possible_actions.append(self.actions["up"])
        if ((state_x < self.grid_size_full - 1)): possible_actions.append(self.actions["down"])
        if ((state_y > 0)): possible_actions.append(self.actions["left"])
        if ((state_y < self.grid_size_full - 1)): possible_actions.append(self.actions["right"])

        possible_actions = np.array(possible_actions, dtype=int)
        return possible_actions

    def get_next_states(self, state, possible_actions):
        # Given a state, what are the posible next states I can reach
        next_state = []
        state_x, state_y = state // self.grid_size_full, state % self.grid_size_full
        for a in possible_actions:
            if a == 0: next_state.append((state_x - 1) * self.grid_size_full + state_y)
            if a == 1: next_state.append(state_x * self.grid_size_full + state_y - 1)
            if a == 2: next_state.append((state_x + 1) * self.grid_size_full + state_y)
            if a == 3: next_state.append(state_x * self.grid_size_full + state_y + 1)
            # if a == 4: next_state.append(state)
        next_state = np.array(next_state, dtype=int)
        return next_state

    def get_initial_distribution(self):
        init_state_neighbouthood = []
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
                init_state_neighbouthood.append(neighbour_state)
        return initial_dist, init_state_neighbouthood

    def place_objects_on_the_grid(self):
        states = self.grid_size_full * self.grid_size_full

        state_object_array = np.zeros((states, self.n_objects))
        states_to_choose_from = np.setdiff1d(range(states), self.init_state_neighbourhood)
        # choose and fix states for each objects randomly
        states_for_object = self.rng.choice(states_to_choose_from, size=self.n_objects_each * self.n_objects, replace=False)
        for i, s in enumerate(states_for_object):
            state_object_array[s, i%self.n_objects] = 1
        return state_object_array, states_for_object


    def int_to_point(self, i):
        """
        Convert a state int into the corresponding coordinate.
        i: State int.
        -> (x, y) int tuple.
        """
        return (i // self.grid_size_full, i % self.grid_size_full)


    def point_to_int(self, x, y):
        """
        Convert a state (x,y) into the corresponding coordinate i.
        x, y: (x, y) int tuple..
        -> state - > int
        """
        return (x * self.grid_size_full + y)

    def get_state_feature_vector_full(self, state):
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
        states = self.grid_size_full * self.grid_size_full
        feature_matrix = np.zeros((self.n_states, self.n_features_full))
        for i in range(states):
            feature_matrix[i, :] = self.get_state_feature_vector_full(i)
        return feature_matrix

    def get_state_feature_matrix(self):
        index_of_reward_features = range(0, self.n_features_reward)
        new_feature_matrix = self.feature_matrix_full[:,index_of_reward_features]
        return new_feature_matrix


    def get_nxn_neighborhood(self, neighborhood):

        outer = list(product(range(-neighborhood, neighborhood+1), range(-neighborhood, neighborhood+1)))
        inner = list(product(range(-neighborhood + 1 , neighborhood), range(-neighborhood + 1, neighborhood)))

        only_outer_neighborhood = [x for x in outer if x not in inner]
        return only_outer_neighborhood

    def draw(self, V, pi, reward, show, strname, fignum):
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
                        #if pi[s, a] == np.max(pi[s, :]):
                        #    pi_[s] = 0.4
                    pi_ = (pi_.reshape(self.grid_size_full, self.grid_size_full))
                    if a == 2:
                        plt.quiver(X, Y, zeros, -pi_, scale=1, units='xy')
                    elif a == 1:
                        plt.quiver(X, Y, -pi_, zeros, scale=1, units='xy')
                    elif a == 0:
                        plt.quiver(X, Y, zeros, pi_, scale=1, units='xy')
                    elif a == 3:
                        plt.quiver(X, Y, pi_, zeros, scale=1, units='xy')
                    # elif a == 4:
                    #     plt.quiver(X, Y, 0, zeros, scale=1, units='dots')

            plt.title(strname + "Opt values and policy")
        if(show == True):
            plt.show()


    def draw_paper(self, V, pi, reward, strname, figsize=(5. / 2.54, 5. / 2.54)):
        """
        Draws figures for the paper.
        """
        plt.figure(figsize=figsize)
        plt.gca().set_aspect('equal')
        if self.terminal_state == 1:
            reward = reward[:-1]
            V = V[:-1]
            pi = pi[:-1]
            states = self.n_states -1
        else:
            states = self.n_states
        reshaped_reward = copy.deepcopy(reward.reshape((self.grid_size_full, self.grid_size_full)))
        reshaped_reward = np.flip(reshaped_reward, 0)
        plt.pcolor(reshaped_reward, cmap='RdBu', vmin=-10., vmax=10.)
        plt.xticks([])
        plt.yticks([])
        plt.gca().tick_params(axis=u'both', which=u'both',length=0)
        plt.gca().set_xticks(range(11), minor=True)
        plt.gca().xaxis.grid(True, which='minor')
        plt.gca().set_yticks(range(11), minor=True)
        plt.gca().yaxis.grid(True, which='minor')
        plt.savefig("%s-reward.pdf" % strname, bbox_inches="tight")
        plt.gcf()

        if V is not None:
            plt.figure(figsize=figsize)
            reshaped_Value = copy.deepcopy(V.reshape((self.grid_size_full, self.grid_size_full)))
            reshaped_Value = np.flip(reshaped_Value, 0)
            plt.pcolor(reshaped_Value, cmap='RdBu', vmin=0., vmax=1./(1.-self.terminal_gamma))
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
                        #if pi[s, a] == np.max(pi[s, :]):
                        #    pi_[s] = 0.4
                    pi_ = (pi_.reshape(self.grid_size_full, self.grid_size_full))
                    if a == 2:
                        plt.quiver(X, Y, zeros, -pi_, scale=1, units='xy')
                    elif a == 1:
                        plt.quiver(X, Y, -pi_, zeros, scale=1, units='xy')
                    elif a == 0:
                        plt.quiver(X, Y, zeros, pi_, scale=1, units='xy')
                    elif a == 3:
                        plt.quiver(X, Y, pi_, zeros, scale=1, units='xy')
                    # elif a == 4:
                    #     plt.quiver(X, Y, 0, zeros, scale=1, units='dots')

            plt.xticks([])
            plt.yticks([])

            plt.savefig("%s-opt-values-and-policy.pdf" % strname, bbox_inches="tight")

    def discrete_matshow(self, data, show, strname):
        # get discrete colormap
        #f = fignum
        #plt.figure(f)
        fig, ax = plt.subplots(figsize=(5. / 2.54, 5. / 2.54))
        colors = ['white', 'black', 'green', 'yellow', 'gray']
        cMap = ListedColormap(colors)
        heatmap = ax.pcolor(data, cmap=cMap, vmin=0, vmax=(len(colors)-1))
        #cmap = plt.get_cmap('gnuplot', np.max(data) - np.min(data) + 1)
        # set limits .5 outside true range
        #mat = plt.matshow(np.flip(data, axis=0), cmap=heatmap, vmin=np.min(data) - .5, vmax=np.max(data) + .5)
        # plt.ylabel("Y")
        # plt.xlabel("X")
        #yticklabelslist = [str(i) for i in np.arange(0, self.grid_size_full, self.grid_size_full//7+1)]
        #yticklabelslist.reverse()
        #plt.yticks(np.arange(0, self.grid_size_full, self.grid_size_full//7+1), yticklabelslist)

        # tell the colorbar to tick at integers
        #cax = plt.colorbar(mat, ticks=np.arange(np.min(data), np.max(data) + 1), orientation='vertical')
        if self.terminal_state == 1:
            states = self.n_states -1
        else:
            states = self.n_states

        x = np.linspace(0, self.grid_size_full - 1, self.grid_size_full) + 0.5
        y = np.linspace(self.grid_size_full - 1, 0, self.grid_size_full) + 0.5
        X, Y = np.meshgrid(x, y)
        zeros = np.zeros((self.grid_size_full, self.grid_size_full))
        #objects = np.flip(self.state_object_array,1)
        objects = self.state_object_array
        # objects = np.flip(objects, 0)
        # objects = np.flip(objects, 1)
        #objects = np.rot90(objects, axes=[0,1])
        if objects is not None:
            for a in range(self.n_objects):
                pi_ = np.zeros(states)
                for s in range(states):
                    if (np.max(objects[s, :]) > 0):
                        pi_[s] = 0.35 * objects[s, a] / np.max(objects[s, :])
                    # if pi[s, a] == np.max(pi[s, :]):
                    #    pi_[s] = 0.4
                pi_ = (pi_.reshape(self.grid_size_full, self.grid_size_full))
                pts = np.vstack(np.where(pi_)).T
                xc = [X[t[0], t[1]] for t in pts]
                yc = [Y[t[0], t[1]] for t in pts]
                if a == 2:
                    #QV2 = plt.quiver(X, Y, zeros, -pi_, scale=1, units='xy')
                    plt.plot(xc, yc, 'k.')
                    #plt.plot(X, Y, zeros, -pi_, scale=1, units='xy')
                elif a == 1:
                    #QV1 = plt.quiver(X, Y, -pi_, zeros, scale=1, units='xy')
                    plt.plot(xc, yc, 'k+')
                elif a == 0:
                    #QV0 = plt.quiver(X, Y, zeros, pi_, scale=1, units='xy')
                    plt.plot(xc, yc, 'k*')
                # elif a == 3:
                #     QV3 = plt.quiver(X, Y, -pi_, zeros, scale=1, units='xy', color='r')
        #plt.quiverkey(QV0, 0.3, 1.1, 0.4, 'O1-UP', coordinates='axes')
        #plt.quiverkey(QV1, 0.5, 1.1, 0.4, 'O2-LE', coordinates='axes')
        #plt.quiverkey(QV2, 0.7, 1.1, 0.4, 'O3-DO', coordinates='axes')


    def draw_objects(self, show, strname):
        state_matrix = np.zeros((self.grid_size_full, self.grid_size_full))

        for s_x in range(self.grid_size_full):
            for s_y in range(self.grid_size_full):
                state = self.point_to_int(s_x, s_y)

                if self.InitD[state] == 1:
                    state_matrix[s_x, s_y] = 1

        state_matrix = np.flip(state_matrix, 0)

        self.discrete_matshow(state_matrix, show, strname)
        plt.gca().set_xticks([], minor=False)
        plt.gca().set_yticks([], minor=False)
        plt.gca().set_xticks(range(11), minor=True)
        plt.gca().xaxis.grid(True, which='minor')
        plt.gca().set_yticks(range(11), minor=True)
        plt.gca().yaxis.grid(True, which='minor')
        plt.gca().tick_params(axis=u'both', which=u'both',length=0)
        plt.gca().set_aspect('equal')
        if (show == True):
            plt.show()



if __name__ == "__main__":

    env_info = {"gridsizefull": 5,
                "theta_e": [0.6, 0.3, 0.1],
                "gamma": 0.99,
                "randomMoveProb": 0.1,
                "terminalState": 1,
                "terminal_gamma": 0.9
                }

    env = Environment(env_info)
    states_with_object_0 = np.where(env.state_object_array[:, 0] == 1)[0]
    states_with_object_1 = np.where(env.state_object_array[:, 1] == 1)[0]
    states_with_object_2 = np.where(env.state_object_array[:, 2] == 1)[0]

    print("states_with_object_0", states_with_object_0)
    print("states_with_object_1", states_with_object_1)
    print("states_with_object_2", states_with_object_2)

    Q, V, _, pi_s = MDPSolver.valueIteration(env, env.reward)
    MDPSolver.computeFeatureSVF_bellmann(env, pi_s)
    # print(env.feature_matrix)
    env.draw_objects(False, "obj")
    env.draw(V, pi_s, env.reward, True, "Plot", 10)
    plt.show()
    exit(0)

    for s in range(env.n_states):
        for a in range(env.n_actions):
            if sum(env.T[s,:, a]) != 1:
                print("fail")
                print(sum(env.T[s,:, a]))
                break
            #print(sum(env.T[s, :, a]))
    env.draw_objects(True, "objects")
    exit(0)
    Q, V, _, pi_s = MDPSolver.valueIteration(env, env.reward)
    MDPSolver.computeFeatureSVF_bellmann(env, pi_s)
    print(env.feature_matrix)
    env.draw_objects(False, "obj")
    env.draw(V, pi_s, env.reward,True, "Plot", 2)
    exit(0)
    states_with_object_0 = np.where(env.state_object_array[:, 0] == 1)[0]
    states_with_object_1 = np.where(env.state_object_array[:, 1] == 1)[0]
    states_with_object_2 = np.where(env.state_object_array[:, 2] == 1)[0]

    print("states_with_object_0", states_with_object_0)
    print("states_with_object_1", states_with_object_1)
    print("states_with_object_2", states_with_object_2)

    print(env.init_state_neighbourhood)
    print(np.where(env.InitD==1))
    print(env.feature_matrix.shape)
    print(env.feature_matrix)
    env.draw_objects(True, "objects")
    exit(0)

    print(env.n_states)
    print(len(env.InitD))
    _, _, _, pi_s = MDPSolver.valueIteration(env, env.reward)
    MDPSolver.computeFeatureSVF_bellmann(env, pi_s)
    print(env.feature_matrix)
    print(env.n_features)
    print(env.n_features_reward)
    env.draw_objects(True, "objects")
    exit(0)
    for s in range(env.n_states):
        for a in range(env.n_actions):
            #print(sum(env.T[s, :, a]))
            if sum(env.T[s,:,a]) < 1 - 1e-16 or 1 + 1e-16 < sum(env.T[s,:,a]):
                print("fail")
                break

    # print(env.T[env.states_for_object,env.n_states-1,:])
    # exit(0)
    states_with_object_0 = np.where(env.state_object_array[:, 0] == 1)[0]
    states_with_object_1 = np.where(env.state_object_array[:, 1] == 1)[0]
    states_with_object_2 = np.where(env.state_object_array[:, 2] == 1)[0]
    print("states_with_object_0", states_with_object_0)
    print("states_with_object_1", states_with_object_1)
    print("states_with_object_2", states_with_object_2)
    print()

    #exit(0)
    print(env.reward)
    print()
    #print(env.feature_matrix.shape)
    print(env.feature_matrix)
    env.draw_objects(True, "objectword")
    exit(0)


    exit(0)
    exit(0)
    print(env.init_state_neighbourhood)
    print(np.where(env.state_object_array[:, 0] == 1))
    print(env.feature_matrix)