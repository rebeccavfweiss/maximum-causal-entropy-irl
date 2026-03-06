"""
CliffWalking environment wrapper for tabular IRL.

Wraps gymnasium's CliffWalking-v0 into the GridEnvironment interface.
The transition matrix is parameterized by a success_rate: with probability
success_rate the intended action is taken, with probability (1-success_rate)
a uniformly random other action is taken instead.

Grid layout (4 rows x 12 cols = 48 states + 1 absorbing terminal):
    Row 0 (top):     states  0-11
    Row 1:           states 12-23
    Row 2:           states 24-35
    Row 3 (bottom):  states 36-47
        Start: 36 (bottom-left)
        Cliff: 37-46
        Goal:  47 (bottom-right)

State encoding: state = row * 12 + col
An artificial absorbing terminal state (index 48) is added.

Features: one-hot encoding over all 49 states (identity matrix).
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import imageio
import copy
import os
from pathlib import Path
from policy import Policy
from environments.environment import GridEnvironment


class CliffWalkingEnvironment(GridEnvironment):
    """
    CliffWalking wrapper for tabular MCE-IRL.

    Parameters
    ----------
    env_args : dict
        theta : list or ndarray — true reward parameters
        gamma : float — discount factor
        success_rate : float — probability intended action succeeds (default 1.0)
        T : int — horizon
    """

    ROWS = 4
    COLS = 12
    N_GYM_STATES = 48  # 4 * 12

    # Gym actions: 0=Up, 1=Right, 2=Down, 3=Left
    ACTION_UP = 0
    ACTION_RIGHT = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3

    START_STATE = 36  # (row=3, col=0)
    GOAL_STATE = 47  # (row=3, col=11)
    CLIFF_STATES = list(range(37, 47))  # (row=3, col=1..10)

    def __init__(self, env_args: dict):
        super().__init__(env_args)

        self.success_rate = env_args.get("success_rate", 1.0)
        self.T = env_args.get("T", 50)
        self.one_hot_features = env_args.get("one_hot_features", True)

        self.actions = {"up": 0, "right": 1, "down": 2, "left": 3}
        self.actions_names = ["up", "right", "down", "left"]
        self.n_actions = 4
        self.rows = self.ROWS
        self.cols = self.COLS

        # 48 grid states + 1 absorbing terminal
        self.n_states = self.N_GYM_STATES + 1
        self.terminal_state = self.N_GYM_STATES  # index 48

        self.n_features = self.n_states if self.one_hot_features else 1

        self.InitD = self._get_initial_distribution()
        self.T_matrix, self.terminal_states = self._compute_transition_matrix()
        self.T_sparse_list = self._compute_transition_sparse_list()
        self.feature_matrix = self._compute_state_feature_matrix()

        if self.one_hot_features:
            # One-hot: feature_matrix @ theta = theta, so reward = theta per state
            self.reward = self.get_reward_for_given_theta(self.theta_reward)
        else:
            # 1D state-index features: reward is theta_reward directly (per-state)
            self.reward = np.array(self.theta_reward, dtype=np.float64)

        self.agent_position = self.START_STATE
        self.time_step = 0

        # Gym env for rendering — use v1 with is_slippery matching our success_rate
        # Gym's is_slippery=True uses 1/3 for intended + 1/3 each perpendicular
        self._gym_is_slippery = abs(self.success_rate - 1.0 / 3.0) < 1e-9
        self._gym_env = gym.make(
            "CliffWalking-v1",
            is_slippery=self._gym_is_slippery,
            render_mode="rgb_array",
        )

    def _get_initial_distribution(self) -> np.ndarray:
        initial_dist = np.zeros(self.n_states)
        initial_dist[self.START_STATE] = 1.0
        return initial_dist

    def _compute_state_feature_matrix(self) -> np.ndarray:
        if self.one_hot_features:
            return np.eye(self.n_states, dtype=np.float64)
        else:
            # 1D features: each state's feature is its integer index
            return np.arange(self.n_states, dtype=np.float64).reshape(-1, 1)

    def _compute_transition_matrix(self) -> tuple[np.ndarray, list[int]]:
        """
        Build transition matrix P[s, s', a] with stochastic actions.

        With probability success_rate, the intended action is executed.
        With probability (1 - success_rate) / 2 each, one of the two
        perpendicular actions is taken instead (no backwards slip).
        This matches gymnasium CliffWalking-v1 is_slippery behaviour.
        """
        terminal_states = [self.terminal_state]

        P = np.zeros((self.n_states, self.n_states, self.n_actions))

        # Absorbing terminal state
        P[self.terminal_state, self.terminal_state, :] = 1.0

        # Cliff and goal states transition to terminal
        for s in self.CLIFF_STATES + [self.GOAL_STATE]:
            P[s, self.terminal_state, :] = 1.0
            terminal_states.append(s)

        slip_prob = (1.0 - self.success_rate) / 2.0

        for s in range(self.N_GYM_STATES):
            if (s in self.CLIFF_STATES) or (s == self.GOAL_STATE):
                continue

            for a in range(self.n_actions):
                # Intended action
                ns = self._deterministic_next_state(s, a)
                P[s, ns, a] += self.success_rate

                # Two perpendicular actions: (a-1)%4 and (a+1)%4
                for perp_a in [(a - 1) % 4, (a + 1) % 4]:
                    ns = self._deterministic_next_state(s, perp_a)
                    P[s, ns, a] += slip_prob

        return P, terminal_states

    def _deterministic_next_state(self, state: int, action: int) -> int:
        """Compute next state for a deterministic action, handling cliff/goal."""
        row = state // self.cols
        col = state % self.cols

        if action == self.ACTION_UP:
            row = max(row - 1, 0)
        elif action == self.ACTION_DOWN:
            row = min(row + 1, self.rows - 1)
        elif action == self.ACTION_LEFT:
            col = max(col - 1, 0)
        elif action == self.ACTION_RIGHT:
            col = min(col + 1, self.cols - 1)

        ns = row * self.cols + col

        # Cliff or goal → absorbing terminal
        if (ns in self.CLIFF_STATES) or (ns == self.GOAL_STATE):
            return self.terminal_state

        return ns

    def reset(self) -> int:
        self.agent_position = self.START_STATE
        self.time_step = 0
        self._gym_env.reset()
        return self.agent_position

    def step(self, action: int) -> tuple[int, float, bool, bool]:
        next_state_prob = self.T_matrix[self.agent_position, :, action]
        new_state = int(np.random.choice(self.n_states, p=next_state_prob))
        self.agent_position = new_state
        self.time_step += 1

        terminated = new_state in self.terminal_states
        return new_state, self.reward[new_state], terminated, self.time_step > self.T

    def state_to_rowcol(self, state: int) -> tuple[int, int]:
        if state >= self.N_GYM_STATES:
            return (-1, -1)
        return (state // self.cols, state % self.cols)

    def rowcol_to_state(self, row: int, col: int) -> int:
        return row * self.cols + col

    def render(
        self,
        policy: Policy,
        T: int = 50,
        store: bool = False,
        reward: np.ndarray = None,
        V: np.ndarray = None,
        show: bool = False,
        strname: str = "",
        fignum: int = 0,
        **kwargs,
    ) -> Path:
        f = fignum
        video_path = None

        if reward is None:
            reward = self.reward

        # Plot reward heatmap (exclude terminal state)
        plt.figure(f)
        reward_grid = reward[: self.N_GYM_STATES].reshape(self.rows, self.cols)
        plt.pcolor(np.flipud(reward_grid))
        plt.colorbar()
        plt.title(f"{strname}: reward function (success_rate={self.success_rate})")
        if show:
            plt.show()
        if store:
            os.makedirs(
                os.path.join("plots", "cliff_walking_environment"), exist_ok=True
            )
            plt.savefig(
                os.path.join(
                    "plots", "cliff_walking_environment", f"{strname}_reward.jpg"
                ),
                format="jpg",
            )
        plt.close()

        # Plot value function + policy arrows
        if V is not None and policy is not None:
            f += 1
            plt.figure(f)
            V_plot = np.flip(V[0, : self.N_GYM_STATES].reshape(self.rows, self.cols), 0)
            # plt.pcolor(np.flipud(V_plot))
            plt.pcolor(V_plot)
            plt.colorbar()

            x = np.linspace(0, self.cols - 1, self.cols) + 0.5
            y = np.linspace(self.rows - 1, 0, self.rows) + 0.5
            X, Y = np.meshgrid(x, y)
            zeros = np.zeros((self.rows, self.cols))

            pi = policy.pi
            current_states = [self.START_STATE]
            visited = []
            for t in range(min(pi.shape[0], T)):
                visited += current_states
                for a in range(self.n_actions):
                    pi_ = np.zeros(self.N_GYM_STATES)
                    for s in current_states:
                        if s < self.N_GYM_STATES and np.max(pi[t, s, :]) > 0:
                            pi_[s] = 0.45 * pi[t, s, a] / np.max(pi[t, s, :])
                    pi_grid = pi_.reshape(self.rows, self.cols)
                    # pi_grid = np.flipud(pi_grid)
                    # Actions: 0=Up, 1=Right, 2=Down, 3=Left
                    if a == self.ACTION_UP:
                        plt.quiver(X, Y, zeros, pi_grid, scale=1, units="xy")
                    elif a == self.ACTION_DOWN:
                        plt.quiver(X, Y, zeros, -pi_grid, scale=1, units="xy")
                    elif a == self.ACTION_LEFT:
                        plt.quiver(X, Y, -pi_grid, zeros, scale=1, units="xy")
                    elif a == self.ACTION_RIGHT:
                        plt.quiver(X, Y, pi_grid, zeros, scale=1, units="xy")

                # Advance to next reachable states
                new_states = set()
                for s in current_states:
                    if s >= self.N_GYM_STATES:
                        continue
                    for a in range(self.n_actions):
                        ns = self._deterministic_next_state(s, a)
                        if ns not in visited and ns < self.N_GYM_STATES:
                            new_states.add(ns)
                current_states = list(new_states)

            plt.title(f"{strname}: value function & policy (sr={self.success_rate})")
            if show:
                plt.show()
            if store:
                plt.savefig(
                    os.path.join(
                        "plots", "cliff_walking_environment", f"{strname}_policy.jpg"
                    ),
                    format="jpg",
                )
            plt.close()

        # Record an episode as MP4 only when our success_rate matches the
        # gym env's dynamics (1.0 deterministic, or 1/3 with is_slippery)
        gym_matches = (self.success_rate == 1.0) or self._gym_is_slippery
        if store and policy is not None and gym_matches:
            video_path = self._record_episode(policy, T, strname)

        return video_path

    def _record_episode(
        self, policy: Policy, T: int, strname: str, fps: int = 2
    ) -> Path:
        """
        Record an episode in the gym environment as MP4.

        Runs the policy in our stochastic wrapper and mirrors the resulting
        trajectory in the gym env for rendering.

        Parameters
        ----------
        policy : Policy
            tabular policy to execute
        T : int
            maximum episode length
        strname : str
            filename prefix
        fps : int
            frames per second for the video

        Returns
        -------
        video_path : Path
            path to the saved MP4 file
        """
        rec_dir = Path("recordings") / "cliff_walking"
        rec_dir.mkdir(parents=True, exist_ok=True)

        # Reset gym env and use its step function directly (only called
        # when our success_rate matches the gym env's dynamics)
        self._gym_env.reset()

        images = []
        img = self._gym_env.render()
        images.append(img)

        gym_state = self.START_STATE
        for t in range(T):
            action = policy.predict(gym_state, t)

            gym_state, _, gym_terminated, gym_truncated, _ = self._gym_env.step(action)
            img = self._gym_env.render()
            images.append(img)

            if gym_terminated or gym_truncated:
                break

        video_path = rec_dir / f"{strname}.mp4"
        imageio.mimsave(
            str(video_path),
            [np.array(img) for img in images],
            fps=fps,
        )

        return video_path
