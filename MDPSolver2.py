
import numpy as np
from abc import ABC, abstractmethod
import copy
import numpy.matlib
import random
import time
from scipy import sparse
import matplotlib.pyplot as plt
import env_objectworld
np.set_printoptions(suppress=True)
np.set_printoptions(precision=12)
#np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=500)

class MDPSolver(ABC):

    @staticmethod
    def valueIteration(env, values, loss=None, tol=1e-6):
        V = np.zeros((env.n_states))
        Q = np.zeros((env.n_states, env.n_actions))
        #reward = env.get_reward_for_given_w(w)
        if loss is None:
            loss=np.zeros(len(values["reward"]))
        values["reward"] += loss

        iter=0
        while True:
            iter +=1
            V_old = copy.deepcopy(V)

            for a in range(env.n_actions):
                Q[:, a] = values["reward"] + env.gamma * env.T_sparse_list[a].dot(V)

            V = np.max(Q, axis=1)

            if abs(np.linalg.norm(V - V_old, np.inf)) < tol:
                break

        # For a deterministic policy
        pi_d = np.argmax(Q, axis=1)

        # For a non-deterministic policy
        #pi_s = np.zeros((env.n_states, env.n_actions))
        pi_s = Q - np.max(Q, axis=1)[:, None]
        #pi_s[np.where(pi_s == 0)] = 1
        pi_s[np.where((-1e-12 <= pi_s) & (pi_s <= 1e-12))] = 1
        pi_s[np.where(pi_s <= 0)] = 0
        pi_s = pi_s/pi_s.sum(axis=1)[:, None]

        return Q, V, pi_d, pi_s

    @abstractmethod
    def soft_valueIteration(env, values, tol=1e-6):
        pass

    def softmax_list(self, A, states):
        Amax = A.max(axis=1)
        Atemp = A - Amax.reshape((states, 1))  # For numerical stability.
        return Amax + \
            np.log(np.exp(Atemp).sum(axis=1))
    
    def computeFeatureSVF_sampling_state(self, env, policy, num_episode, len_episode, init_state=None):
        # Given a policy, return feature expectation and state-visitation frequency
        # Step1: Generate "m" trajectories of that policy (sampling)
        # Step2: Compute the feature expectation using Equation5 of the paper
        # But, if policy is deterministic, num_episode = 1
        episode_list = []
        mu_list = []
        sv_list = []
        if len(policy.shape) == 1:
            num_episode = 1

        feature_expectation = np.zeros(env.n_states)
        total_state_count_gamma = np.zeros(env.n_states)
        for i in range(num_episode):
            episode, state_count_gamma = self.generateEpisode(env, policy, len_episode, init_state)
            total_state_count_gamma += state_count_gamma
            episode_list.append(episode)
            mu_list.append(env.get_state_feature_matrix().transpose().dot(state_count_gamma))
            sv_list.append(state_count_gamma)
        feature_expectation = env.get_state_feature_matrix().transpose().dot(total_state_count_gamma)

        feature_variance = feature_expectation.dot(feature_expectation.transpose())


        # return both state visitation frequency and feature expectation
        return total_state_count_gamma/num_episode, feature_expectation/num_episode, feature_variance/num_episode, episode_list, mu_list, sv_list


    def computeFeatureSVF_sampling(self, env, policy, num_episode, len_episode):
        # Given a policy, return feature expectation and state-visitation frequency
        # Step1: Generate "m" trajectories of that policy (sampling)
        # Step2: Compute the feature expectation using Equation5 of the paper
        # But, if my policy is deterministic, num_episode = 1
        if len(policy.shape) == 1:
            num_episode = 1
        feature_expectation = np.zeros(env.n_states)
        total_state_count_gamma = np.zeros(env.n_states)
        for i in range(num_episode):
            _, state_count_gamma = self.generateEpisode(env, policy, len_episode)
            total_state_count_gamma += state_count_gamma
        feature_expectation = env.get_state_feature_matrix().transpose().dot(total_state_count_gamma)
        feature_variance = feature_expectation.dot(feature_expectation.transpose())
        # return both state visitation frequency and feature expectation
        return total_state_count_gamma/num_episode, feature_expectation/num_episode, feature_variance/num_episode


    def generateEpisode(self, env, policy, len_episode, init_state=None):
        # Given a policy, returns state_visitation counts (gamma times) for one trajectory
        # First, lets check our policy, if its deterministic lets convert to stochastic
        if len(policy.shape) == 1:
            changed_policy = self.convert_det_to_stochastic_policy(env, policy)
        else:
            changed_policy = policy
        state_counts_gamma = np.zeros(env.n_states)
        # Selecting a start state according to InitD
        state = int(np.random.choice(np.arange(env.n_states), p=env.InitD))
        if init_state is not None:
            state = init_state

        episode = []
        for t in range(len_episode):  # length of episodes we are generating
            episode.append(state)
            state_counts_gamma[state] += env.gamma ** t
            if env.terminal_state == 1 and state == env.n_states - 1:
                break
            probs = changed_policy[state]
            action = int(np.random.choice(np.arange(len(probs)), p=probs))
            next_state = env.T[state, :, action]
            state = int(np.random.choice(np.arange(env.n_states), p=next_state))

        return episode, state_counts_gamma
    
    @staticmethod
    def convert_det_to_stochastic_policy(env, deterministicPolicy):
        # Given a deterministic Policy return a stochastic policy
        stochasticPolicy = np.zeros((env.n_states, env.n_actions))
        for i in range(env.n_states):
            stochasticPolicy[i][deterministicPolicy[i]] = 1
        return stochasticPolicy

    @staticmethod
    def get_T_pi(env, policy):
        T_pi = np.zeros((env.n_states, env.n_states))
        for n_s in range(env.n_states):
            for a in range(env.n_actions):
                T_pi[:, n_s] += policy[:, a] * env.T[:, n_s, a]

        return T_pi
    
    @staticmethod
    def computeFeatureSVF_bellmann_averaged(env, policy, num_iter=None):
        # To ensure stochastic behaviour in the feature expectation and state-visitation frequencies (run num_iter times)
        # But, if input policy is deterministic, set num_iter = 1
        if len(policy.shape) == 1:
            num_iter = 1
        if (num_iter == None):
            num_iter = 1

        sv_list = []
        mu_list = []
        nu_list = []
        for i in range(num_iter):
            SV, feature_expectation, feature_variance = MDPSolver.computeFeatureSVF_bellmann(env, policy)
            sv_list.append(SV)
            mu_list.append(feature_expectation)
            nu_list.append(feature_variance)

        return np.mean(sv_list, axis=0), np.mean(mu_list, axis=0), np.mean(nu_list, axis=0)
    
    @staticmethod
    def computeFeatureSVF_bellmann(env, policy, tol=1e-6):
        # Given a policy, Return State Visitation Frequencies and feature Expectation
        # Using Bellman Equation
        # Let's ensure we have a stochastic policy, if not lets convert
        if len(policy.shape) == 1:
            changed_policy = MDPSolver.convert_det_to_stochastic_policy(env, policy)
        else:
            changed_policy = policy
        # Creating a T matrix for the policy

        T_pi = MDPSolver.get_T_pi(env, changed_policy)

        if env.terminal_state == 1:
            T_pi[-1, :] = 0

        # Converting T to a sparse matrix
        #start = time.time()
        T_pi_sparse = sparse.csr_matrix(T_pi.transpose())
        #print("time to create sparse matrix in svf_bellmann=", time.time() - start)
        # Some initialisations
        SV = np.zeros((env.n_states))
        iter = 0
        # Bellman Equation
        while True:
            iter += 1
            SV_old = copy.deepcopy(SV)
            SV[:] = env.InitD + env.gamma * T_pi_sparse.dot(SV[:])
            if abs(np.linalg.norm(SV - SV_old)) < tol:
                break
        # Converged Now. Return both SV and feature expectation
        feature_expectation = env.get_state_feature_matrix().transpose().dot(SV)

        feature_variance = np.matmul(feature_expectation.reshape(len(feature_expectation),1),feature_expectation.reshape(len(feature_expectation),1).transpose())
 
        return SV, feature_expectation, feature_variance
    
    def computeValueFunction_bellmann_averaged(self, env, policy, values, num_iter=None):
        # To ensure stochastic behavior, run it multiple times and take an expectation
        # But, if policy is deterministic set num_iter to 1
        if len(policy.shape) == 1:
            num_iter = 1
        if (num_iter == None):
            num_iter = 1

        V_list = []
        for i in range(num_iter):
            V_list.append(self.computeValueFunction_bellmann(env, policy, values))
        return np.mean(V_list, axis=0)
    
    @abstractmethod
    def computeValueFunction_bellmann(self, env, policy, values, tol=1e-6):
        pass



class MDPSolverExpectation(MDPSolver):

    def soft_valueIteration(self, env, values, tol=1e-6):
        V = np.zeros((env.n_states))
        Q = np.zeros((env.n_states, env.n_actions))
        #reward = env.get_reward_for_given_w(w)

        iter=0
        while True:
            iter +=1
            V_old = copy.deepcopy(V)

            for a in range(env.n_actions):
                Q[:, a] = values["reward"] + env.gamma * env.T_sparse_list[a].dot(V)
            #V = np.max(Q, axis=1)
            V = self.softmax_list(Q, env.n_states)

            if abs(np.linalg.norm(V - V_old, np.inf)) < tol:
                break

        temp = copy.deepcopy(Q)

        # Softmax by row to interpret these values as probabilities.
        temp -= temp.max(axis=1).reshape((env.n_states, 1))  # For numerical stability.
        pi_s = np.exp(temp) / np.exp(temp).sum(axis=1).reshape((env.n_states, 1))

        return Q, V, pi_s

    def computeValueFunction_bellmann(self, env, policy, values, tol=1e-6):
        # Given a policy (could be either deterministic or stochastic), I return the Value Function
        # Using the Bellman Equations
        # Let's check if this policy is deterministic or stochastic
        if len(policy.shape) == 1:
            changed_policy = self.convert_det_to_stochastic_policy(env, policy)
        else:
            changed_policy = policy

        T_pi = self.get_T_pi(env, changed_policy)

        # Converting this T to a sparse matrix
        T_pi_sparse = sparse.csr_matrix(T_pi)
        # Some more initialisations
        V = np.zeros((env.n_states))
        #reward = env.get_reward_for_given_w(w)
        iter = 0
        # Bellman Equation
        while True:
            iter += 1
            V_old = copy.deepcopy(V)
            V[:] = values["reward"] + env.gamma * T_pi_sparse.dot(V)
            if abs(np.linalg.norm(V - V_old, np.inf)) < tol: 
                #converged
                break
        return V
        


class MDPSolverVariance(MDPSolver):

    def soft_valueIteration(self, env, values, tol=1e-6):
        V = np.zeros((env.n_states))
        Q = np.zeros((env.n_states, env.n_actions))

        iter=0
        while True:
            iter +=1
            V_old = copy.deepcopy(V)

            for a in range(env.n_actions):
                #not sure if that actually works due to dependence on t
                Q[:, a] = values["reward"] + values["variance"] +  env.gamma * env.T_sparse_list[a].dot(V)

            V = self.softmax_list(Q, env.n_states)

            if abs(np.linalg.norm(V - V_old, np.inf)) < tol:
                break

        temp = copy.deepcopy(Q)

        # Softmax by row to interpret these values as probabilities.
        temp -= temp.max(axis=1).reshape((env.n_states, 1))  # For numerical stability.
        pi_s = np.exp(temp) / np.exp(temp).sum(axis=1).reshape((env.n_states, 1))

        return Q, V, pi_s

    def computeValueFunction_bellmann(self, env, policy, values, tol=1e-6):
            # Given a policy (could be either deterministic or stochastic), I return the Value Function
            # Using the Bellman Equations
            # Let's check if this policy is deterministic or stochastic
            if len(policy.shape) == 1:
                changed_policy = self.convert_det_to_stochastic_policy(env, policy)
            else:
                changed_policy = policy

            T_pi = self.get_T_pi(env, changed_policy)

            # Converting this T to a sparse matrix
            T_pi_sparse = sparse.csr_matrix(T_pi)
            # Some more initialisations
            V = np.zeros((env.n_states))
            #reward = env.get_reward_for_given_w(w)
            iter = 0
            # Bellman Equation
            while True:
                iter += 1
                V_old = copy.deepcopy(V)
                V[:] = values["reward"] + values["variance"] + env.gamma * T_pi_sparse.dot(V)
                if abs(np.linalg.norm(V - V_old, np.inf)) < tol: 
                    #converged
                    break
            return V









if __name__ == "__main__":
    ######################################## NOTES
    # For sampling, num episodes = 200, and len episode = 5/(1-\gamma)
    # For SVF computation using Bellmann, iter=1 is enough, unless there is lot of stochasticity in the policy
    # Bellmann with iter=3 is still faster than sampling for grid-size <= 100
    # Bellmann with iter=1 is equal to sampling for grid-size = 128
    ########################################
    env_info = {"gridsizefull": 3,
                "gridsizemacro":1,
                "initialStateDistNonzero": -1, ## ignored when debugFlag=1
                "binaryRewardFlag": 0, ## ignored when debugFlag=1
                "nonzeroRewards": 2, ## ignored when debugFlag=1
                "randomMoveProb": 0,
                "gamma": 0.99,
                "terminalState": 1,
                "terminalGamma": 1,
                "debugFlag": 3
                }

    start = time.time()
    env = env_objectworld.Environment(env_info)



    print("grid size=", env.grid_size_full, "num of states=", env.n_states)
    print("Hmax=", env.Hmax)
    print("time to create environment=", time.time() - start)

    print("Grid size length: ", env.grid_size_full, " Macro size length: ", env.grid_size_macro, " Total States: ", env.n_states, "Total Features (Macros): ", env.n_features)

    print(env.T[0, :, 0])
    print(env.T[:, :, 1])
    print(env.T[:, :, 2])
    print(env.T[:, :, 3])
    exit(1)

    ######################################## compute policy and plot
    print("\n-------------------------------")
    start = time.time()
    Vy, pi_dy, pi_sy = valueIteration(env, env.w_star)
    #print("value iteration V= \n", Vy )
    print("time to run value iteration=", time.time() - start)

    print("\n-------------------------------")
    start = time.time()
    Vy, pi_s = soft_valueIteration(env, env.w_star)
    #print("Soft_value iteration V= \n", Vy )
    print("time to run value iteration=", time.time() - start)

    ######################################## compute policy and plot


    print("\n-------------------------------")
    start = time.time()
    Vy, pi_dy, pi_sy = valueIteration(env, env.w_star)
    print("time to run value iteration=", time.time() - start)

    V = computeValueFunction_bellmann(env, pi_dy, env.w_star)
    print("This is V computed using Bellman Function Given a Policy:\n", V)
    print("This is V computed using Value Iteration:\n", Vy)

    env.draw(Vy, pi_sy, env.w_star, show=False, strname="Teacher - ", fignum=1)
    #env.draw(Vy, convert_det_to_stochastic_policy(env, pi_dy), env.w_star, show=False, strname="Teacher - ", fignum=3)

    print("pi_sy=\n", pi_sy)
    print("pi_dy=\n", pi_dy)

    ######################################## debugging sampling vs. Bellmann
    start = time.time()
    svf_sampling, mu_sampling = computeFeatureSVF_sampling(env, pi_sy, 100, 300) # num episodes, length of episiode
    print("svf_sampling=\n", svf_sampling)
    print("mu_sampling=\n", mu_sampling)
    print("time to compute mu_sampling=", time.time() - start)

    start = time.time()
    svf_bellman, mu_bellman = computeFeatureSVF_bellmann_averaged(env, pi_sy, 5)
    print("svf_bellman=\n", svf_bellman)
    print("mu_bellman=\n", mu_bellman)
    print("time to compute mu_bellman   =", time.time() - start)
    print("norm of svf values: ", np.linalg.norm(svf_bellman - svf_sampling))
    print("norm of mu values: ", np.linalg.norm(mu_bellman - mu_sampling))
    # print(env.format_pi(pi_dy))

    plt.show()