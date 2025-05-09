import numpy as np
import copy
from scipy import sparse
from MDP_solver import MDPSolver
from environments.environment import Environment
from policy import Policy, TabularPolicy

np.set_printoptions(suppress=True)
np.set_printoptions(precision=12)
np.set_printoptions(linewidth=500)


class MDPSolverExact(MDPSolver):
    """
    abstract class collecting the basic methods a MDP solver needs
    For this solver we expect the environment to be small and easy to access, i.e., number of states and an explicit index on the observations is available
    If this cannot be guaranteed then use MDPSolverApproximation instead.

    Parameters
    ----------
    T : int
        finite horizon value
    compute_variance : bool
            whether or not variance term should be computed (for efficiency reasons will only be computed if necessary)
    """

    def __init__(self, T: int, compute_variance: bool):
        super().__init__(T, compute_variance)


    def softmax_list(self, A: np.ndarray, n_states: int) -> np.ndarray:
        """
        computes the softmax of a given matrix column wise

        Parameters
        ----------
        A : ndarray
            matrix to compute the softmax of
        n_states : int
            number of states

        Returns
        -------
        ndarray of numerically stable softmax values
        """
        Amax = A.max(axis=1)
        Atemp = A - Amax.reshape((n_states, 1))  # For numerical stability.
        return Amax + np.log(np.exp(Atemp).sum(axis=1))

    def get_T_pi(self, env: Environment, policy: Policy) -> np.ndarray:
        """
        computes state transition probability based on a policy

        Parameters
        ----------
        env : environment.Environment
            the environment representing the setting of the problem
        policy : ndarray
            policy to use

        Returns
        -------
        T_pi : ndarray
            transition probability matrix based on the policy (time dependent)
        """
        assert isinstance(policy, TabularPolicy)
        T_pi = np.zeros((self.T, env.n_states, env.n_states))

        for t in range(self.T):
            for n_s in range(env.n_states):
                for a in range(env.n_actions):
                    T_pi[t, :, n_s] += policy.pi[t, :, a] * env.T_matrix[:, n_s, a]

        return T_pi
    
    def get_T_pi_from_trajectory(self, env: Environment, trajectory: list[tuple[int,int,int,float]]) -> np.ndarray:
        """
        computes state transition probability based on a fixed trajectory (treated as deterministic policy)

        Parameters
        ----------
        env : environment.Environment
            the environment representing the setting of the problem
        trajectory : list[tuple[int,int,int, float]]
            predefined trajectory of an agent containing tuples with: current state, action, next state, reward

        Returns
        -------
        T_pi : ndarray
            transition probability matrix based on the trajectory (time dependent)
        """
        T_pi = np.zeros((self.T, env.n_states, env.n_states))

        length = min(len(trajectory), self.T)

        for i in range(length):
            T_pi[i,trajectory[i][0], trajectory[i][2]] = 1.0

        return T_pi

    
    def compute_feature_SVF_bellmann(
        self, env: Environment, policy: Policy, trajectory:list[tuple[int,int,int,float]]=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        computes feature SVF

        Parameters
        ----------
        env : environment.Environment
            the environment representing the setting of the problem
        policy : ndarray
            policy to use (time dependent)
        trajectory : list[tuple[int,int,int, float]]
            predefined trajectory of an agent

        Returns
        -------
        feature_expectation : ndarray
        feature_variance : ndarray
        """

        SV = self.compute_SV(env, policy, trajectory)

        feature_matrix = env.get_state_feature_matrix()

        feature_expectation = sum(
            feature_matrix.transpose().dot(SV[t]) for t in range(self.T)
        )

        feature_variance = np.zeros((env.n_features, env.n_features))

        if self.compute_variance:

                # Compute feature_products using batch matrix multiplication
            feature_products = np.einsum("ai,bj->abij", feature_matrix, feature_matrix)

            # Compute variance using vectorized operations
            for t1 in range(self.T):
                for t2 in range(self.T):
                    if t1 == t2:
                        feature_variance += np.einsum(
                            "s,ijs->ij",
                            SV[t1],
                            feature_products.diagonal(axis1=0, axis2=1),
                        )
                    else:
                        feature_variance += np.einsum(
                            "s,t,stij->ij", SV[t1], SV[t2], feature_products
                        )

        return feature_expectation, feature_variance

    def compute_SV(self, env:Environment, policy:Policy, trajectory:list[tuple[any,int,any, float]]=None) -> np.ndarray:
        """
        Computes State value counts

        Parameters
        ----------
        env : Environment
            environment to use
        policy : Policy
            policy to use
        trajectory : list[tuple[any, int, any, float]]
            if we should treat a given trajectory like a deterministic policy instead

        Returns
        -------
        SV : ndarray
            state value counts
        """
        if trajectory is None:
            # Creating a T matrix for the policy
            T_pi = self.get_T_pi(env, policy)
        else: 
            T_pi = self.get_T_pi_from_trajectory(env, trajectory)

        for s in env.terminal_states:
            T_pi[:, s, :] = 0.0

        T_pi_sparse = [sparse.csr_matrix(T_pi[t].transpose()) for t in range(self.T)]

        SV = np.zeros((self.T, env.n_states))

        # Bellman Equation
        SV[0, :] = env.InitD
        for t in range(1, self.T):
            SV[t, :] = env.gamma * T_pi_sparse[t - 1].dot(SV[t - 1, :])
        return SV

    def compute_value_function_bellmann_averaged(
        self, env: Environment, policy, values: dict[str:any], num_iter: int = None
    ) -> np.ndarray:
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
        if num_iter == None:
            num_iter = 1

        V_list = []

        for _ in range(num_iter):
            V_list.append(self.compute_value_function_bellmann(env, policy, values))

        return np.mean(V_list, axis=0)

    def compute_value_function_bellmann(
        self, env: Environment, policy: np.ndarray, values: dict[str:any]
    ) -> np.ndarray:
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

        T_pi = self.get_T_pi(env, policy)

        T_pi_sparse = [sparse.csr_matrix(T_pi[t]) for t in range(self.T)]

        V = np.zeros((self.T, env.n_states))

        # Bellman Equation
        V[self.T - 1, :] = values["reward"]
        for t in range(self.T - 2, -1, -1):
            V[t, :] = values["reward"] + env.gamma * T_pi_sparse[t].dot(V[t + 1, :])

        return V


class MDPSolverExactExpectation(MDPSolverExact):
    """
    MDP solver that uses feature expectation matching

    Parameters
    ----------
    T : int
        finite horizon value
    compute_variance : bool
            whether or not variance term should be computed (for efficiency reasons will only be computed if necessary) (default = False)
    """

    def __init__(self, T: int = 45, compute_variance: bool = False):
        super().__init__(T, compute_variance)

    def soft_value_iteration(
        self, env: Environment, values: dict[str:any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        V[self.T - 1, :] = self.softmax_list(Q[self.T - 1, :, :], env.n_states)

        for t in range(self.T - 2, -1, -1):
            for a in range(env.n_actions):
                Q[t, :, a] = values["reward"] + env.gamma * env.T_sparse_list[a].dot(
                    V[t + 1]
                )

            V[t, :] = self.softmax_list(Q[t, :, :], env.n_states)

        temp = copy.deepcopy(Q)
        for t in range(self.T):
            for s in range(env.n_states):
                temp[t, s, :] -= V[t, s]

        pi_s = np.zeros((self.T, env.n_states, env.n_actions))

        for t in range(self.T):
            # Softmax by row to interpret these values as probabilities.
            temp[t, :, :] -= (
                temp[t, :, :].max(axis=1).reshape((env.n_states, 1))
            )  # For numerical stability.
            pi_s[t, :, :] = np.exp(temp[t, :, :]) / np.exp(temp[t, :, :]).sum(
                axis=1
            ).reshape((env.n_states, 1))

        for s in env.terminal_states:
            pi_s[:, s, :] = 0.0

        return Q, V, pi_s


class MDPSolverExactVariance(MDPSolverExact):
    """
    MDP solver that uses feature expectation and variance matching (using recurive evaluation as finite horizon)

    Parameters
    ----------
    T : int
        finite horizon value
    compute_variance : bool
            whether or not variance term should be computed (for efficiency reasons will only be computed if necessary) (default = True)
    """

    def __init__(self, T: int = 45, compute_variance: bool = True):
        super().__init__(T, compute_variance)

    def soft_value_iteration(
        self, env: Environment, values: dict[str:any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            Q[self.T - 1, :, a] = (
                values["reward"] + env.gamma ** (self.T - 1) * values["variance"]
            )

        V[self.T - 1, :] = self.softmax_list(Q[self.T - 1, :, :], env.n_states)

        for t in range(self.T - 2, -1, -1):
            for a in range(env.n_actions):
                Q[t, :, a] = (
                    values["reward"]
                    + env.gamma**t * values["variance"]
                    + env.gamma * env.T_sparse_list[a].dot(V[t + 1])
                )

            V[t, :] = self.softmax_list(Q[t, :, :], env.n_states)

        temp = copy.deepcopy(Q)
        for t in range(self.T):
            for s in range(env.n_states):
                temp[t, s, :] -= V[t, s]

        pi_s = np.zeros((self.T, env.n_states, env.n_actions))

        for t in range(self.T):
            # Softmax by row to interpret these values as probabilities.
            temp[t, :, :] -= (
                temp[t, :, :].max(axis=1).reshape((env.n_states, 1))
            )  # For numerical stability.
            pi_s[t, :, :] = np.exp(temp[t, :, :]) / np.exp(temp[t, :, :]).sum(
                axis=1
            ).reshape((env.n_states, 1))

        for s in env.terminal_states:
            pi_s[:, s, :] = 0.0

        return Q, V, pi_s
