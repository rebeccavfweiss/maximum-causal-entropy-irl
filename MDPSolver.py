import numpy as np
from abc import ABC, abstractmethod
import copy
import time
from scipy import sparse
import matplotlib.pyplot as plt
from env_objectworld import Environment

np.set_printoptions(suppress=True)
np.set_printoptions(precision=12)
np.set_printoptions(linewidth=500)

class MDPSolver(ABC):
    """
    abstract class collecting the basic methods a MDP solver needs
    """

    @staticmethod
    def valueIteration(env:Environment, values:dict[str:any], tol:float=1e-6):
        """
        implements value iteration

        Parameters
        ----------
        env : env_objectworld.Environment
            the environment representing the setting of the problem
        values: dict[str: any]
            dictionary storing the reward and variance terms for general use
        tol : float
            convergence tolerance
        
        Returns
        -------
        Q : ndarray
            Q function
        V : ndarray
            value function
        pi_d : ndarray
            deterministic policy
        pi_s : ndarray
            stochastic policy
        """

        V = np.zeros((env.n_states))
        Q = np.zeros((env.n_states, env.n_actions))

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
        pi_s = Q - np.max(Q, axis=1)[:, None]
        pi_s[np.where((-1e-12 <= pi_s) & (pi_s <= 1e-12))] = 1
        pi_s[np.where(pi_s <= 0)] = 0
        pi_s = pi_s/pi_s.sum(axis=1)[:, None]

        return Q, V, pi_d, pi_s

    @abstractmethod
    def soft_valueIteration(env:Environment, values:dict[str:any], tol:float=1e-6):
        pass

    def softmax_list(self, A, states):
        """
        computes the softmax of a given matrix column wise

        Parameters
        ----------
        A
            matrix to compute the softmax of
        state
            number of states
        
        Returns
        -------
        ndarray of stable softmax values
        """
        Amax = A.max(axis=1)
        Atemp = A - Amax.reshape((states, 1))  # For numerical stability.
        return Amax + \
            np.log(np.exp(Atemp).sum(axis=1))
    
    def computeFeatureSVF_sampling_state(self, env:Environment, policy, num_episode:int, len_episode:int, init_state:int=None):
        """
        computes feature SVF

        Given a policy, return feature expectation and state-visitation frequency
        - Step1: Generate "m" trajectories of that policy (sampling)
        - Step2: Compute the feature expectation and variance

        But, if policy is deterministic, num_episode = 1

        Parameters
        ----------
        env : env_objectworld.Environment
            the environment representing the setting of the problem
        policy : ndarray ??
            policy to use
        num_episode : int
            number of episodes to be generated
        len_episode : int
            episode length
        init_state : int
            state to use as starting state

        Returns
        -------
        state visitation frequency : ndarray
        feature_expectation : ndarray
        feature_variance : ndarray
        """

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

        # return both state visitation frequency and feature expectation, variance
        return total_state_count_gamma/num_episode, feature_expectation/num_episode, feature_variance/num_episode, episode_list, mu_list, sv_list


    def computeFeatureSVF_sampling(self, env:Environment, policy, num_episode:int, len_episode:int):
        """
        computes feature SVF

        Given a policy, return feature expectation and state-visitation frequency
        - Step1: Generate "m" trajectories of that policy (sampling)
        - Step2: Compute the feature expectation and variance

        But, if policy is deterministic, num_episode = 1

        Parameters
        ----------
        env : env_objectworld.Environment
            the environment representing the setting of the problem
        policy : ndarray ??
            policy to use
        num_episode : int
            number of episodes to be generated
        len_episode : int
            episode length

        Returns
        -------
        state visitation frequency : ndarray
        feature_expectation : ndarray
        feature_variance : ndarray
        """
        if len(policy.shape) == 1:
            num_episode = 1
        feature_expectation = np.zeros(env.n_states)
        total_state_count_gamma = np.zeros(env.n_states)
        for i in range(num_episode):
            _, state_count_gamma = self.generateEpisode(env, policy, len_episode)
            total_state_count_gamma += state_count_gamma

        feature_expectation = env.get_state_feature_matrix().transpose().dot(total_state_count_gamma)
        feature_variance = feature_expectation.dot(feature_expectation.transpose())

        # return both state visitation frequency and feature expectation, variance
        return total_state_count_gamma/num_episode, feature_expectation/num_episode, feature_variance/num_episode


    def generateEpisode(self, env:Environment, policy, len_episode:int, init_state:int=None):
        """
        generates an episode in the given setting

        Parameters
        ----------
        env : env_objectworld.Environment
            the environment representing the setting of the problem
        policy : ndarray ??
            policy to use
        len_episode : int
            episode length
        init_state : int
            state to use as starting state

        Returns
        -------
        episode : list[int]
            list of visited states within the episode
        state_counts_gamma : ndarray
            discounted state visitation counts for one trajectory
        """

        # check policy, if its deterministic convert to stochastic
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
    def convert_det_to_stochastic_policy(env:Environment, deterministicPolicy):
        """
        converts a deterministic policy to a stochastic one

        Parameters
        ----------
        env : env_objectworld.Environment
            the environment representing the setting of the problem
        deterministicPolicy
            policy to convert

        Returns
        -------
        stochasticPolicy : ndarray
            converted policy
        """
        # Given a deterministic Policy return a stochastic policy
        stochasticPolicy = np.zeros((env.n_states, env.n_actions))
        for i in range(env.n_states):
            stochasticPolicy[i][deterministicPolicy[i]] = 1
        return stochasticPolicy

    @staticmethod
    def get_T_pi(env:Environment, policy):
        """
        computes state transition probability based on a policy

        Parameters
        ----------
        env : env_objectworld.Environment
            the environment representing the setting of the problem
        policy
            policy to use

        Returns
        -------
        T_pi : ndarray
            transition probability matrix based on the policy
        """
        T_pi = np.zeros((env.n_states, env.n_states))
        for n_s in range(env.n_states):
            for a in range(env.n_actions):
                T_pi[:, n_s] += policy[:, a] * env.T[:, n_s, a]

        return T_pi
    
    @staticmethod
    def computeFeatureSVF_bellmann_averaged(env:Environment, policy, num_iter:int=None):
        """
        computes averages feature SVF

        Parameters
        ----------
        env : env_objectworld.Environment
            the environment representing the setting of the problem
        policy : ??
            policy to use
        num_iter : int
            number of iterations

        Returns
        -------
        mean state visitation frequencies
        mean feature expectation
        mean feature variance


        """
        # To ensure stochastic behaviour in the feature expectation and state-visitation frequencies (run num_iter times)
        # But, if input policy is deterministic, set num_iter = 1
        if len(policy.shape) == 1:
            num_iter = 1
        if (num_iter == None):
            num_iter = 1

        sv_list = []
        mu_list = []
        nu_list = []
        for _ in range(num_iter):
            SV, feature_expectation, feature_variance = MDPSolver.computeFeatureSVF_bellmann(env, policy)
            sv_list.append(SV)
            mu_list.append(feature_expectation)
            nu_list.append(feature_variance)

        return np.mean(sv_list, axis=0), np.mean(mu_list, axis=0), np.mean(nu_list, axis=0)
    
    @staticmethod
    def computeFeatureSVF_bellmann(env:Environment, policy, tol:float=1e-6):
        """
        computes feature SVF

        Parameters
        ----------
        env : env_objectworld.Environment
            the environment representing the setting of the problem
        policy : ??
            policy to use
        tol : float
            convergence tolerance

        Returns
        -------
        SV : ndarray
        feature_expectation : ndarray
        feature_variance : ndarray
        """
        # Using Bellman Equation
        # ensure stochastic policy
        if len(policy.shape) == 1:
            changed_policy = MDPSolver.convert_det_to_stochastic_policy(env, policy)
        else:
            changed_policy = policy

        # Creating a T matrix for the policy
        T_pi = MDPSolver.get_T_pi(env, changed_policy)

        if env.terminal_state == 1:
            T_pi[-1, :] = 0

        # Converting T to a sparse matrix

        T_pi_sparse = sparse.csr_matrix(T_pi.transpose())

        SV = np.zeros((env.n_states))
        iter = 0

        # Bellman Equation
        while True:
            iter += 1
            SV_old = copy.deepcopy(SV)
            SV[:] = env.InitD + env.gamma * T_pi_sparse.dot(SV[:])
            if abs(np.linalg.norm(SV - SV_old)) < tol:
                break

        feature_expectation = env.get_state_feature_matrix().transpose().dot(SV)
        feature_variance = np.matmul(feature_expectation.reshape(len(feature_expectation),1),feature_expectation.reshape(len(feature_expectation),1).transpose())
 
        return SV, feature_expectation, feature_variance
    
    def computeValueFunction_bellmann_averaged(self, env:Environment, policy, values:dict[str:any], num_iter:int=None):
        """
        compute averages value function using bellmann equations
        
        Parameters
        ----------
        env : env_objectworld.Environment
            the environment representing the setting of the problem
        policy : ??
            policy to use
        values : dict[str:any]
            dictionary with the feature expecation and variance term needed for bellmann
        num_iter : int
            number of iterations

        Returns
        -------
        V_list : ndarray
            mean value function
        """
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
    """ 
    MDP solver that uses feature expectation matching
    """

    def soft_valueIteration(self, env:Environment, values:dict[str:any], tol:float=1e-6):
        """
        computes soft value iteration using feature expectation matching

        Parameters
        ----------
        env : env_objectworld.Environment
            the environment representing the setting of the problem
        values : dict[str:any]
            dictionary with the feature expecation and variance term needed for bellmann
        tol : float
            convergence tolerance

        Returns
        -------
        Q : ndarray
            state-action value function
        V : ndarray
            state value function
        pi_s : ndarray
            stochastic policy
        """
        V = np.zeros((env.n_states))
        Q = np.zeros((env.n_states, env.n_actions))

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

    def computeValueFunction_bellmann(self, env:Environment, policy, values:dict[str:any], tol:float=1e-6):
        """
        computes value function using bellmann equations
        
        Parameters
        ----------
        env : env_objectworld.Environment
            the environment representing the setting of the problem
        policy : ??
            policy to use (can be deterministic or stochastic)
        values : dict[str:any]
            dictionary with the feature expecation and variance term needed for bellmann
        tol : float
            convergence tolerance

        Returns
        -------
        V : ndarray
            value function
        """
        # check if policy is deterministic or stochastic
        if len(policy.shape) == 1:
            changed_policy = self.convert_det_to_stochastic_policy(env, policy)
        else:
            changed_policy = policy

        T_pi = self.get_T_pi(env, changed_policy)

        # Converting this T to a sparse matrix
        T_pi_sparse = sparse.csr_matrix(T_pi)

        V = np.zeros((env.n_states))

        iter = 0
        # Bellman Equation
        while True:
            iter += 1
            V_old = copy.deepcopy(V)
            V[:] = values["reward"] + env.gamma * T_pi_sparse.dot(V)
            if abs(np.linalg.norm(V - V_old, np.inf)) < tol: 
                break

        return V
        

class MDPSolverVariance(MDPSolver):
    """
    MDP solver that uses feature expectation and variance matching
    """

    def soft_valueIteration(self, env:Environment, values:dict[str:any], tol:float=1e-6):
        """
        computes soft value iteration using feature expectation and variance matching

        Parameters
        ----------
        env : env_objectworld.Environment
            the environment representing the setting of the problem
        values : dict[str:any]
            dictionary with the feature expecation and variance term needed for bellmann
        tol : float
            convergence tolerance

        Returns
        -------
        Q : ndarray
            state-action value function
        V : ndarray
            state value function
        pi_s : ndarray
            stochastic policy
        """
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

    def computeValueFunction_bellmann(self, env:Environment, policy, values:dict[str:any], tol:float=1e-6):
        """
        computes value function using bellmann equations
        
        Parameters
        ----------
        env : env_objectworld.Environment
            the environment representing the setting of the problem
        policy : ??
            policy to use (can be deterministic or stochastic)
        values : dict[str:any]
            dictionary with the feature expecation and variance term needed for bellmann
        tol : float
            convergence tolerance

        Returns
        -------
        V : ndarray
            value function
        """

        # check if policy is deterministic or stochastic
        if len(policy.shape) == 1:
            changed_policy = self.convert_det_to_stochastic_policy(env, policy)
        else:
            changed_policy = policy

        T_pi = self.get_T_pi(env, changed_policy)

        # Converting this T to a sparse matrix
        T_pi_sparse = sparse.csr_matrix(T_pi)
        
        V = np.zeros((env.n_states))
        
        iter = 0
        # Bellman Equation
        while True:
            iter += 1
            V_old = copy.deepcopy(V)
            V[:] = values["reward"] + values["variance"] + env.gamma * T_pi_sparse.dot(V)
            if abs(np.linalg.norm(V - V_old, np.inf)) < tol: 
                break

        return V