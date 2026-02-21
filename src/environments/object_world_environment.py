"""
Implementation of the ObjectWorld benchmark environment from Levine, Sergey, Popovic, Zoran, and Koltun, Vladlen. Non-
linear inverse reinforcement learning with gaussian processes.

Implementation adapted from <https://github.com/TroddenSpade/Maximum-Entropy-Deep-IRL/blob/main/envs/ObjectWorld.py>
"""

import numpy as np
from itertools import product
import math
from environments.environment import GridEnvironment
from pathlib import Path
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from IPython import display
from policy import Policy
import os
import copy

plt.rcParams["animation.ffmpeg_path"] = (
    "C:\\Users\\rebec\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.0.1-full_build\\bin\\ffmpeg.exe"
)


class WorldObject(object):
    def __init__(self, inner_color, outer_color):
        self.inner_color = inner_color
        self.outer_color = outer_color


class ObjectWorldEnvironment(GridEnvironment):
    """
    Class wrapper for the object world benchmark environment

    Parameters
    ----------
    env_args : dict[Any]
        environment definition parameters depending on the specific environment used
    """

    def __init__(self, env_args: dict):
        super().__init__(env_args)

        self.wind = float(env_args.get("wind", 0.3))
        self.grid_size = env_args["grid_size"]
        self.actions = {"up": 0, "left": 1, "down": 2, "right": 3, "nothing": 4}
        self.actions_names = ["up", "left", "down", "right", "nothing"]
        self.action_deltas = {
            "up": (0, 1),
            "left": (-1, 0),
            "down": (0, -1),
            "right": (1, 0),
            "nothing": (0, 0),
        }
        self.n_actions = len(self.actions)
        self.n_states = self.grid_size**2
        self.n_objects = env_args.get("n_objects", 15)
        self.n_colors = env_args.get("n_colors", 3)
        self.random_start = env_args["random_start"]
        self.discrete = not env_args.get("continuous", False)
        self.T = env_args["T"]

        self.InitD = self._get_initial_distribution()

        self.n_features = (
            2 * self.n_colors * self.grid_size if self.discrete else 2 * self.n_colors
        )

        self.objects = {}
        for _ in range(self.n_objects):
            obj = WorldObject(
                np.random.randint(self.n_colors), np.random.randint(self.n_colors)
            )
            while True:
                x = np.random.randint(self.grid_size)
                y = np.random.randint(self.grid_size)

                if (x, y) not in self.objects:
                    break
            self.objects[x, y] = obj

        print(self.objects)

        self.T_matrix, self.terminal_states = self._compute_transition_matrix()
        self.T_sparse_list = self._compute_transition_sparse_list()
        self.feature_matrix = self._compute_state_feature_matrix()
        self.reward = np.array([self._reward(s) for s in range(self.n_states)])

        self.agent_position = int(
            np.random.choice(np.arange(self.n_states), p=self.InitD)
        )

        self.time_step = 0

    def _get_initial_distribution(self):

        if self.random_start:

            return 1 / self.n_states * np.ones(self.n_states)

        else:
            initial_distr = np.zeros(self.n_states)
            initial_distr[0] = 1.0
            return initial_distr

    def _compute_transition_matrix(self):
        """
        Computes the transition matrix for this environment and find the terminal states

        Returns
        -------
        P : ndarray
            transition matrix
        terminal_states : list
            terminal states of the environment
        """

        terminal_state = []

        dynamics = np.zeros((self.n_states, self.n_states, self.n_actions))
        # S_t+1, A_t, S_t
        for s in range(self.n_states):
            x, y = s % self.grid_size, s // self.grid_size
            for a in self.actions:
                x_a, y_a = self.action_deltas[a]
                for d in self.actions:
                    x_d, y_d = self.action_deltas[d]
                    if 0 <= x + x_d < self.grid_size and 0 <= y + y_d < self.grid_size:
                        dynamics[
                            s, (x + x_d) + (y + y_d) * self.grid_size, self.actions[a]
                        ] += (self.wind / self.n_actions)
                    else:
                        dynamics[s, s, self.actions[a]] += self.wind / self.n_actions
                if 0 <= x + x_a < self.grid_size and 0 <= y + y_a < self.grid_size:
                    dynamics[
                        s, (x + x_a) + (y + y_a) * self.grid_size, self.actions[a]
                    ] += (1 - self.wind)
                else:
                    dynamics[s, s, self.actions[a]] += 1 - self.wind

        # there are no terminal states in that sense, we just cut off after a certain amount of time
        return dynamics, terminal_state

    def _compute_state_feature_matrix(self) -> np.ndarray:
        """
        Returns
        -------
        feature_matrix : ndarray
            representing full feature matrix
        """
        feature_matrix = np.zeros((self.n_states, self.n_features))
        for i in range(self.n_states):
            feature_matrix[i, :] = self.__get_state_feature_vector_full(i)
        return feature_matrix

    def __get_state_feature_vector_full(self, state: int, discrete=True) -> np.ndarray:
        """
        Returns
        -------
        feature_vector : ndarray
            represents the features of the given state
        """

        x_s, y_s = state % self.grid_size, state // self.grid_size

        nearest_inner = {}
        nearest_outer = {}

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if (x, y) in self.objects:
                    dist = math.hypot((x - x_s), (y - y_s))
                    obj = self.objects[x, y]
                    if obj.inner_color in nearest_inner:
                        if dist < nearest_inner[obj.inner_color]:
                            nearest_inner[obj.inner_color] = dist
                    else:
                        nearest_inner[obj.inner_color] = dist
                    if obj.outer_color in nearest_outer:
                        if dist < nearest_outer[obj.outer_color]:
                            nearest_outer[obj.outer_color] = dist
                    else:
                        nearest_outer[obj.outer_color] = dist

        for c in range(self.n_colors):
            if c not in nearest_inner:
                nearest_inner[c] = 0
            if c not in nearest_outer:
                nearest_outer[c] = 0

        if discrete:
            state = np.zeros((2 * self.n_colors * self.grid_size,))
            i = 0
            for c in range(self.n_colors):
                for d in range(1, self.grid_size + 1):
                    if nearest_inner[c] < d:
                        state[i] = 1
                    i += 1
                    if nearest_outer[c] < d:
                        state[i] = 1
                    i += 1
        else:
            state = np.zeros((2 * self.n_colors))
            i = 0
            for c in range(self.n_colors):
                state[i] = nearest_inner[c]
                i += 1
                state[i] = nearest_outer[c]
                i += 1

        return state

    def _reward(self, state_p):
        x, y = state_p % self.grid_size, state_p // self.grid_size

        near_c0 = False
        near_c1 = False
        for dx, dy in product(range(-3, 4), range(-3, 4)):
            if 0 <= x + dx < self.grid_size and 0 <= y + dy < self.grid_size:
                if (
                    abs(dx) + abs(dy) <= 3
                    and (x + dx, y + dy) in self.objects
                    and self.objects[x + dx, y + dy].outer_color == 0
                ):
                    near_c0 = True
                if (
                    abs(dx) + abs(dy) <= 2
                    and (x + dx, y + dy) in self.objects
                    and self.objects[x + dx, y + dy].outer_color == 1
                ):
                    near_c1 = True
        if near_c0 and near_c1:
            return 1
        if near_c0:
            return -1
        return 0

    def reset(self) -> any:
        """
        Reset wrapper to generalize environment access over different environments

        Returns
        -------
        Initial state description
        """
        self.agent_position = int(
            np.random.choice(np.arange(self.n_states), p=self.InitD)
        )

        self.time_step = 0
        return self.agent_position

    def step(self, action: int) -> tuple[any, float, bool, bool]:
        """
        Step wrapper to generalize environment access over different environments

        Parameters
        ----------
        action : int
            action to take

        Returns
        -------
        new_state
            current state description
        reward : float
            reward for taken action
        terminated : bool
            if episode is terminated
        truncated : bool
            if episode was truncated
        """
        next_state_prob = self.T_matrix[self.agent_position, :, action]
        new_state = int(np.random.choice(self.n_states, p=next_state_prob))
        self.agent_position = new_state
        self.time_step += 1

        return (
            new_state,
            self.reward[new_state],
            new_state in self.terminal_states,
            self.time_step > self.T,
        )

    def __int_to_point(self, i: int) -> tuple[int, int]:
        """
        Returns
        -------
        tuple[int,int] :  representing the coordinate for the given state
        """
        return (i % self.grid_size, i // self.grid_size)

    def render(
        self,
        rewards: list,
        V: list,
        policy: Policy,
        strname: str = "",
        store: bool = False,
        show: bool = True,
        fignum: int = 0,
        **kwargs,
    ) -> None:
        """Visualize the learning of the reward function across training

        Parameters
        ----------
        rewards: list
            list of rewards per iteration
        V:
            value function
        strname : str
            file name to store
        store : bool
            whether the visualization should be stored
        show: bool
            whether the visualization should be shown
        fignum : int
            figure identifier
        """
        f = fignum
        if len(rewards) == 1:
            figure = plt.figure(f)

            plt.pcolor(rewards[0].reshape(self.grid_size, self.grid_size))

            plt.colorbar()
            plt.title(f"{strname} reward")
            if show:
                plt.show()
            if store:
                plt.savefig(
                    Path("plots")
                    / "object_world_environment"
                    / f"{strname}_rewards.jpg",
                    format="jpg",
                )
            plt.close()

        if V is not None:

            x = np.linspace(0, self.grid_size - 1, self.grid_size) + 0.5
            y = np.linspace(self.grid_size - 1, 0, self.grid_size) + 0.5
            X, Y = np.meshgrid(x, y)
            zeros = np.zeros((self.grid_size, self.grid_size))
            f += 1
            plt.figure(f)
            V_plot = V[0, :]
            reshaped_Value = copy.deepcopy(
                V_plot.reshape((self.grid_size, self.grid_size))
            )
            # reshaped_Value = np.flip(reshaped_Value, 0)
            plt.pcolor(reshaped_Value)  # , vmin=-10)
            plt.colorbar()
            if policy.pi is not None:
                current_states = [0]
                visited = []
                for t in range(policy.pi.shape[0]):

                    visited += current_states
                    for a in range(self.n_actions):
                        pi_ = np.zeros(self.n_states)
                        for s in current_states:
                            if np.max(policy.pi[t, s, :]) > 0:
                                pi_[s] = (
                                    0.45
                                    * policy.pi[t, s, a]
                                    / np.max(policy.pi[t, s, :])
                                )

                        pi_ = pi_.reshape(self.grid_size, self.grid_size)
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
                            for x in self.__get_next_states(
                                n, np.array(list(range(5)), dtype=int)
                            )
                            if x not in visited
                        )
                    )

            plt.title(strname + ": optimal values and policy")
            if show:
                plt.show()
            if store:
                plt.savefig(
                    Path("plots")
                    / "object_world_environment"
                    / f"{strname}_policy.jpg",
                    format="jpg",
                )

            plt.close()

        if len(rewards) > 1:
            if show or store:
                figure = plt.figure(f + 1)
                ax = plt.subplot()

                def AnimationFunction(frame, skip_factor):
                    if frame % 10 == 0:
                        print(
                            f"Rendering frame {skip_factor*frame}/{len(rewards)}...",
                            end="\r",
                        )

                    ax.pcolor(
                        rewards[skip_factor * frame].reshape(
                            self.grid_size, self.grid_size
                        ),
                    )

                skip_factor = max(1, int(len(rewards) / 50))
                anim_created = FuncAnimation(
                    figure,
                    lambda frame: AnimationFunction(frame, skip_factor),
                    frames=int(len(rewards) / skip_factor),
                    interval=25,
                )

                if show:
                    video = anim_created.to_html5_video()
                    html = display.HTML(video)
                    display.display(html)

                if store:
                    writer = FFMpegWriter(
                        fps=30, metadata=dict(artist="Me"), bitrate=1800
                    )
                    anim_created.save(
                        Path("recordings") / "object_world" / f"rewards_{strname}.mp4",
                        writer=writer,
                    )

                plt.close()

    def __get_next_states(self, state: int, possible_actions: np.ndarray) -> np.ndarray:
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
        state_x, state_y = state % self.grid_size, state // self.grid_size
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
                if state_x < self.grid_size - 1:
                    n_state_x = n_state_x + 1
            if a == 3:
                if state_y < self.grid_size - 1:
                    n_state_y = n_state_y + 1

            next_state.append(n_state_y * self.grid_size + n_state_x)

        next_state = np.array(next_state, dtype=int)
        return next_state
