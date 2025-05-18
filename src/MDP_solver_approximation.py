import numpy as np
import copy
from MDP_solver import MDPSolver
from environments.environment import Environment
from policy import Policy

np.set_printoptions(suppress=True)
np.set_printoptions(precision=12)
np.set_printoptions(linewidth=500)

class MDPSolverApproximation(MDPSolver):
    """
    abstract class collecting the basic methods a MDP solver needs
    For this solver we expect the environment to be large w.r.t the observation space
    If the environment is, e.g., an easy gridworld with a known enumeration of the states then use MDPSolverExact instead as it will be more accurate.

    Parameters
    ----------
    T : int
        finite horizon value
    compute_variance : bool
            whether or not variance term should be computed (for efficiency reasons will only be computed if necessary)
    """

    def __init__(self, T: int, compute_variance: bool):
        super().__init__(T, compute_variance)

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
            not used here but given for uniform interface
        trajectory : list[tuple[any,int,int, float]]
            predefined trajectory of an agent

        Returns
        -------
        feature_expectation : ndarray
        feature_variance : ndarray
        """

        feature_sum = trajectory[0][0] + sum(env.gamma**(t+1)*trajectory[t][2] for t in range(len(trajectory)))

        feature_sum = feature_sum.astype(np.float32)
        feature_sum_prod = np.outer(feature_sum, feature_sum)


        return feature_sum, feature_sum_prod
    
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
