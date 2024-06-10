import numpy as np
from abc import ABC, abstractmethod
import copy
from scipy import sparse
from environment import Environment

np.set_printoptions(suppress=True)
np.set_printoptions(precision=12)
np.set_printoptions(linewidth=500)


class MDPSolver(ABC):
    """
    abstract class collecting the basic methods a MDP solver needs

    Parameters
    ----------
    T : int
        finite horizon value
    """

    def __init__(self, T: int = 10):
        self.T = T

    def valueIteration(self, env: Environment, values: dict[str:any]):
        """
        implements value iteration for finite horizon

        Parameters
        ----------
        env : environment.Environment
            the environment representing the setting of the problem
        values: dict[str: any]
            dictionary storing the reward and variance terms for general use

        Returns
        -------
        Q : ndarray
            Q function
        V : ndarray
            value function
        pi_d : ndarray
            deterministic policy (time dependent)
        pi_s : ndarray
            stochastic policy (time dependent)
        """

        V = np.zeros((self.T, env.n_states))
        Q = np.zeros((self.T, env.n_states, env.n_actions))

        for a in range(env.n_actions):
            Q[self.T - 1, :, a] = values["reward"]

        V[self.T - 1, :] = np.max(Q[self.T - 1, :, :], axis=1)

        for t in range(self.T - 2, -1, -1):
            for a in range(env.n_actions):
                Q[t, :, a] = values["reward"] + env.gamma * env.T_sparse_list[a].dot(
                    V[t + 1]
                )

            V[t, :] = np.max(Q[t, :, :], axis=1)

        # For a deterministic policy
        pi_d = np.zeros((self.T, env.n_states))

        for t in range(self.T):
            for s in range(env.n_states):
                pi_d[t][s] = max(range(env.n_actions), key=lambda a: Q[t][s][a])

        # For a non-deterministic policy
        pi_s = np.zeros((self.T, env.n_states, env.n_actions))

        for t in range(self.T):
            pi_s[t, :, :] = Q[t, :, :] - np.max(Q[t, :, :], axis=1)[:, None]
            pi_s[t, :, :][
                np.where((-1e-12 <= pi_s[t, :, :]) & (pi_s[t, :, :] <= 1e-12))
            ] = 1
            pi_s[t, :, :][np.where(pi_s[t, :, :] <= 0)] = 0
            pi_s[t, :, :] = pi_s[t, :, :] / pi_s[t, :, :].sum(axis=1)[:, None]

        return Q, V, pi_d, pi_s

    @abstractmethod
    def soft_valueIteration(env: Environment, values: dict[str:any]):
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
        ndarray of numerically stable softmax values
        """
        Amax = A.max(axis=1)
        Atemp = A - Amax.reshape((states, 1))  # For numerical stability.
        return Amax + np.log(np.exp(Atemp).sum(axis=1))

    def computeFeatureSVF_sampling_state(
        self,
        env: Environment,
        policy,
        num_episode: int,
        len_episode: int,
        init_state: int = None,
    ):
        """
        computes feature SVF

        Given a policy, return feature expectation and state-visitation frequency
        - Step1: Generate "num_episode" trajectories of that policy (sampling)
        - Step2: Compute the feature expectation and variance

        if policy is deterministic, num_episode = 1

        Parameters
        ----------
        env : environment.Environment
            the environment representing the setting of the problem
        policy : ndarray
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

        if len(policy.shape) == 2:
            num_episode = 1

        feature_expectation = np.zeros(env.n_states)
        total_state_count_gamma = np.zeros(env.n_states)

        for _ in range(num_episode):
            episode, state_count_gamma = self.generateEpisode(
                env, policy, len_episode, init_state
            )
            total_state_count_gamma += state_count_gamma
            episode_list.append(episode)
            mu_list.append(
                env.get_state_feature_matrix().transpose().dot(state_count_gamma)
            )
            sv_list.append(state_count_gamma)

        feature_expectation = (
            env.get_state_feature_matrix().transpose().dot(total_state_count_gamma)
        )

        feature_variance = feature_expectation.dot(feature_expectation.transpose())

        return (
            total_state_count_gamma / num_episode,
            feature_expectation / num_episode,
            feature_variance / num_episode,
            episode_list,
            mu_list,
            sv_list,
        )

    def computeFeatureSVF_sampling(
        self, env: Environment, policy, num_episode: int, len_episode: int
    ):
        """
        computes feature SVF

        Given a policy, return feature expectation and state-visitation frequency
        - Step1: Generate "m" trajectories of that policy (sampling)
        - Step2: Compute the feature expectation and variance

        But, if policy is deterministic, num_episode = 1

        Parameters
        ----------
        env : environment.Environment
            the environment representing the setting of the problem
        policy : ndarray
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
        if len(policy.shape) == 2:
            num_episode = 1

        feature_expectation = np.zeros(env.n_states)
        total_state_count_gamma = np.zeros(env.n_states)

        for _ in range(num_episode):
            _, state_count_gamma = self.generateEpisode(env, policy, len_episode)
            total_state_count_gamma += state_count_gamma

        feature_expectation = (
            env.get_state_feature_matrix().transpose().dot(total_state_count_gamma)
        )
        feature_variance = feature_expectation.dot(feature_expectation.transpose())

        return (
            total_state_count_gamma / num_episode,
            feature_expectation / num_episode,
            feature_variance / num_episode,
        )

    def generateEpisode(
        self, env: Environment, policy, len_episode: int, init_state: int = None
    ):
        """
        generates an episode in the given setting

        Parameters
        ----------
        env : environment.Environment
            the environment representing the setting of the problem
        policy : ndarray
            policy to use
        len_episode : int
            episode length
        init_state : int
            state to use as starting state (default = None)

        Returns
        -------
        episode : list[int]
            list of visited states within the episode
        state_counts_gamma : ndarray
            discounted state visitation counts for one trajectory
        """

        # check policy, if its deterministic convert to stochastic
        if len(policy.shape) == 2:
            changed_policy = self.convert_det_to_stochastic_policy(env, policy)
        else:
            changed_policy = policy

        state_counts_gamma = np.zeros(env.n_states)
        # Selecting a start state according to InitD
        state = int(np.random.choice(np.arange(env.n_states), p=env.InitD))

        if init_state is not None:
            state = init_state

        episode = []
        for t in range(len_episode):
            episode.append(state)
            state_counts_gamma[state] += env.gamma**t
            if (state == env.n_states - 1) or (t == self.T):
                break
            probs = changed_policy[t, state]
            action = int(np.random.choice(np.arange(len(probs)), p=probs))
            next_state = env.T[state, :, action]
            state = int(np.random.choice(np.arange(env.n_states), p=next_state))

        return episode, state_counts_gamma

    def convert_det_to_stochastic_policy(self, env: Environment, deterministicPolicy):
        """
        converts a deterministic policy to a stochastic one

        Parameters
        ----------
        env : environment.Environment
            the environment representing the setting of the problem
        deterministicPolicy
            policy to convert (time dependent)

        Returns
        -------
        stochasticPolicy : ndarray
            converted policy
        """

        stochasticPolicy = np.zeros((self.T, env.n_states, env.n_actions))

        for t in range(self.T):
            for i in range(env.n_states):
                stochasticPolicy[t][i][deterministicPolicy[t, i]] = 1

        return stochasticPolicy

    def get_T_pi(self, env: Environment, policy):
        """
        computes state transition probability based on a policy

        Parameters
        ----------
        env : environment.Environment
            the environment representing the setting of the problem
        policy
            policy to use

        Returns
        -------
        T_pi : ndarray
            transition probability matrix based on the policy (time dependent)
        """
        T_pi = np.zeros((self.T, env.n_states, env.n_states))

        for t in range(self.T):
            for n_s in range(env.n_states):
                for a in range(env.n_actions):
                    T_pi[t, :, n_s] += policy[t, :, a] * env.T[:, n_s, a]

        return T_pi

    def computeFeatureSVF_bellmann_averaged(
        self, env: Environment, policy, num_iter: int = None
    ):
        """
        computes average feature SVF

        Parameters
        ----------
        env : environment.Environment
            the environment representing the setting of the problem
        policy : ndarray
            policy to use (time dependent)
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
        if len(policy.shape) == 2:
            num_iter = 1
        if num_iter == None:
            num_iter = 1

        sv_list = []
        mu_list = []
        nu_list = []

        for _ in range(num_iter):
            SV, feature_expectation, feature_variance = self.computeFeatureSVF_bellmann(
                env, policy
            )
            sv_list.append(SV)
            mu_list.append(feature_expectation)
            nu_list.append(feature_variance)

        return (
            np.mean(sv_list, axis=0),
            np.mean(mu_list, axis=0),
            np.mean(nu_list, axis=0),
        )

    def computeFeatureSVF_bellmann(self, env: Environment, policy):
        """
        computes feature SVF

        Parameters
        ----------
        env : environment.Environment
            the environment representing the setting of the problem
        policy : ndarray
            policy to use (time dependent)

        Returns
        -------
        SV : ndarray
        feature_expectation : ndarray
        feature_variance : ndarray
        """

        # ensure stochastic policy
        if len(policy.shape) == 2:
            changed_policy = self.convert_det_to_stochastic_policy(env, policy)
        else:
            changed_policy = policy

        # Creating a T matrix for the policy
        T_pi = self.get_T_pi(env, changed_policy)
        T_pi[:, -1, :] = 0

        T_pi_sparse = [sparse.csr_matrix(T_pi[t].transpose()) for t in range(self.T)]

        SV = np.zeros((self.T, env.n_states))

        # Bellman Equation
        SV[0, :] = env.InitD
        for t in range(1, self.T):
            SV[t, :] = env.gamma * T_pi_sparse[t].dot(SV[t - 1, :])

        feature_expectation = sum(
            env.get_state_feature_matrix().transpose().dot(SV[t]) for t in range(self.T)
        )
        feature_variance = np.matmul(
            feature_expectation.reshape(len(feature_expectation), 1),
            feature_expectation.reshape(len(feature_expectation), 1).transpose(),
        )

        return np.sum(SV, axis=0), feature_expectation, feature_variance

    def computeValueFunction_bellmann_averaged(
        self, env: Environment, policy, values: dict[str:any], num_iter: int = None
    ):
        """
        compute averages value function using bellmann equations

        Parameters
        ----------
        env : environment.Environment
            the environment representing the setting of the problem
        policy : ndarray
            policy to use (time dependent)
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
        if len(policy.shape) == 2:
            num_iter = 1

        if num_iter == None:
            num_iter = 1

        V_list = []

        for _ in range(num_iter):
            V_list.append(self.computeValueFunction_bellmann(env, policy, values))

        return np.mean(V_list, axis=0)

    @abstractmethod
    def computeValueFunction_bellmann(
        self, env: Environment, policy, values: dict[str:any]
    ):
        pass


class MDPSolverExpectation(MDPSolver):
    """
    MDP solver that uses feature expectation matching

    Parameters
    ----------
    T : int
        finite horizon value
    """

    def __init__(self, T: int):
        super().__init__(T)

    def soft_valueIteration(self, env: Environment, values: dict[str:any]):
        """
        computes soft value iteration using feature expectation matching (using recurive evaluation as finite horizon)

        Parameters
        ----------
        env : environment.Environment
            the environment representing the setting of the problem
        values : dict[str:any]
            dictionary with the feature expecation and variance term needed for bellmann

        Returns
        -------
        Q : ndarray
            state-action value function
        V : ndarray
            state value function
        pi_s : ndarray
            stochastic policy
        """
        V = np.zeros((self.T, env.n_states))
        Q = np.zeros((self.T, env.n_states, env.n_actions))

        for a in range(env.n_actions):
            Q[self.T - 1, :, a] = values["reward"]

        V[self.T - 1, :] = np.max(Q[self.T - 1, :, :], axis=1)

        for t in range(self.T - 2, -1, -1):
            for a in range(env.n_actions):
                Q[t, :, a] = values["reward"] + env.gamma * env.T_sparse_list[a].dot(
                    V[t + 1]
                )

            V[t, :] = self.softmax_list(Q[t, :, :], env.n_states)

        temp = copy.deepcopy(Q)

        pi_s = np.zeros((self.T, env.n_states, env.n_actions))

        for t in range(self.T):
            # Softmax by row to interpret these values as probabilities.
            temp[t, :, :] -= (
                temp[t, :, :].max(axis=1).reshape((env.n_states, 1))
            )  # For numerical stability.
            pi_s[t, :, :] = np.exp(temp[t, :, :]) / np.exp(temp[t, :, :]).sum(
                axis=1
            ).reshape((env.n_states, 1))

        return Q, V, pi_s

    def computeValueFunction_bellmann(
        self, env: Environment, policy, values: dict[str:any]
    ):
        """
        computes value function using bellmann equations

        Parameters
        ----------
        env : environment.Environment
            the environment representing the setting of the problem
        policy : ndarray
            policy to use (can be deterministic or stochastic)
        values : dict[str:any]
            dictionary with the feature expecation and variance term needed for bellmann


        Returns
        -------
        V : ndarray
            value function
        """
        # check if policy is deterministic or stochastic
        if len(policy.shape) == 2:
            changed_policy = self.convert_det_to_stochastic_policy(env, policy)
        else:
            changed_policy = policy

        T_pi = self.get_T_pi(env, changed_policy)

        T_pi_sparse = [sparse.csr_matrix(T_pi[t]) for t in range(self.T)]

        V = np.zeros((self.T, env.n_states))

        # Bellman Equation
        V[self.T - 1, :] = values["reward"]
        for t in range(self.T - 2, -1, -1):
            V[t, :] = values["reward"] + env.gamma * T_pi_sparse[t].dot(V[t + 1, :])

        return V


class MDPSolverVariance(MDPSolver):
    """
    MDP solver that uses feature expectation and variance matching (using recurive evaluation as finite horizon)

    Parameters
    ----------
    T : int
        finite horizon value
    """

    def __init__(self, T: int):
        super().__init__(T)

    def soft_valueIteration(self, env: Environment, values: dict[str:any]):
        """
        computes soft value iteration using feature expectation and variance matching

        Parameters
        ----------
        env : environment.Environment
            the environment representing the setting of the problem
        values : dict[str:any]
            dictionary with the feature expecation and variance term needed for bellmann

        Returns
        -------
        Q : ndarray
            state-action value function
        V : ndarray
            state value function
        pi_s : ndarray
            stochastic policy
        """
        V = np.zeros((self.T, env.n_states))
        Q = np.zeros((self.T, env.n_states, env.n_actions))

        for a in range(env.n_actions):
            Q[self.T - 1, :, a] = values["reward"]

        V[self.T - 1, :] = np.max(Q[self.T - 1, :, :], axis=1)

        for t in range(self.T - 2, -1, -1):
            for a in range(env.n_actions):
                Q[t, :, a] = (
                    values["reward"]
                    + env.gamma**t * values["variance"]
                    + env.gamma * env.T_sparse_list[a].dot(V[t + 1])
                )

            V[t, :] = self.softmax_list(Q[t, :, :], env.n_states)

        temp = copy.deepcopy(Q)

        pi_s = np.zeros((self.T, env.n_states, env.n_actions))

        for t in range(self.T):
            # Softmax by row to interpret these values as probabilities.
            temp[t, :, :] -= (
                temp[t, :, :].max(axis=1).reshape((env.n_states, 1))
            )  # For numerical stability.
            pi_s[t, :, :] = np.exp(temp[t, :, :]) / np.exp(temp[t, :, :]).sum(
                axis=1
            ).reshape((env.n_states, 1))

        return Q, V, pi_s

    def computeValueFunction_bellmann(
        self, env: Environment, policy, values: dict[str:any]
    ):
        """
        computes value function using bellmann equations

        Parameters
        ----------
        env : environment.Environment
            the environment representing the setting of the problem
        policy : ndarray
            policy to use (can be deterministic or stochastic)
        values : dict[str:any]
            dictionary with the feature expecation and variance term needed for bellmann

        Returns
        -------
        V : ndarray
            value function
        """

        # check if policy is deterministic or stochastic
        if len(policy.shape) == 2:
            changed_policy = self.convert_det_to_stochastic_policy(env, policy)
        else:
            changed_policy = policy

        T_pi = self.get_T_pi(env, changed_policy)

        # Converting this T to a sparse matrix
        T_pi_sparse = [sparse.csr_matrix(T_pi[t]) for t in range(self.T)]

        V = np.zeros((self.T, env.n_states))

        # Bellman Equation
        V[self.T - 1, :] = values["reward"]
        for t in range(self.T - 2, -1, -1):
            V[t, :] = (
                values["reward"]
                + env.gamma**t * values["variance"]
                + env.gamma * T_pi_sparse[t].dot(V[t + 1, :])
            )

        return V
