import numpy as np
from abc import abstractmethod
import gymnasium as gym
import minigrid
from policy import Policy
import imageio
from environments.environment import Environment


class MinigridEnvironment(Environment):
    """
    Environment class exposing desired functionality from gymnasium.Environments and adding necessary information for training in these environments

    Parameters
    ----------
    env_args: dict[Any]
        environment definition parameters depending on the specific environment used
    """

    def __init__(self, env_args: dict):

        super().__init__(env_args)

        self.env = gym.make(
            env_args["env_name"],
            render_mode=env_args["render_mode"],
            size=env_args["grid_size"],
        )

        # to always produce the same grid when reset() is called
        self.seed = env_args["seed"]
        # inelegant way to move around unwanted wrappers to get to raw data structures
        self.env = self.env.env.env
        if env_args["use_CoordStateWrapper"]:
            self.env = CrossingCoordStateWrapper(self.env, seed = self.seed)        

        self.n_actions = self.env.action_space.n

        self.T_matrix, self.terminal_states = self._compute_transition_matrix()
        self.T_sparse_list = self._compute_transition_sparse_list()
        self.feature_matrix = self._compute_state_feature_matrix()

        self.InitD = self._get_initial_distribution()

        self.width = self.env.width
        self.height = self.env.height
        self.n_states = self.feature_matrix.shape[0]
        self.n_features = self.feature_matrix.shape[1]

        self.reward = self._compute_true_reward()

    def reset(self) -> any:
        """
        Reset wrapper to generalize environment access over different environments

        Returns
        -------
        Initial state description
        """

        state, _ = self.env.reset(self.seed)

        return state

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
        new_state, reward, terminated, truncated, _ = self.env.step(action)

        return new_state, reward, terminated, truncated

    def render(
        self, policy: Policy, T: int = 20, store:bool=False, strname: str = "", fps: int = 1, **kwargs
    ) -> None:
        """
        Function to record a video of the given policy in the environment

        Parameters
        ----------
        pi : ndarray
            policy to use
        T : int
            maximal episode length
        store : bool
            whether to store the rendering
        strname : str
            file name to store
        fps : int
            frames per second
        """

        images = []
        terminated = False
        truncated = False
        state = self.reset()
        img = self.env.render()
        images.append(img)
        t = 0
        while (not (terminated or truncated)) and t < T:
            # Take the action (index) that have the maximum expected future reward given that state
            action = policy.predict(state, t)
            state, _, terminated, truncated = self.step(action)
            img = self.env.render()
            images.append(img)
            t += 1
        if store:
            imageio.mimsave(
                f"recordings\{strname}.mp4", [np.array(img) for i, img in enumerate(images)], fps=fps
            )

    def action_sample(self) -> int:
        """
        Action space sample wrapper to generalize access over different environments

        Returns
        -------
        action : int
            random action to take
        """
        return self.env.action_space.sample()

    @abstractmethod
    def _compute_true_reward(self):
        pass


class CrossingMiniGridEnvironment(MinigridEnvironment):
    """
    Class to work specifically with the MiniGrid Crossing environment

    Parameters
    ----------
    env_args: dict[Any]
        environment definition parameters depending on the specific environment used
    """

    def __init__(self, env_args: dict):
        self.n_orientations = 4
        env_args["use_CoordStateWrapper"] = True
        super().__init__(env_args)
        # overwrite so agent does not try actions that are unused for better efficiency
        self.n_actions = 3

    def _compute_state_feature_matrix(self) -> np.ndarray:
        """
        Returns
        -------
        feature_matrix : ndarray
            representing full feature matrix
        """

        feature_matrix = np.zeros((self.n_states, 6))
        grid = self.env.env.grid

        for n in range(self.n_states-1):
            x, y, _ = self.env.from_state_index(n)
            feature_matrix[n][0] = abs(x - self.env.goal_position[0]) / self.env.width
            feature_matrix[n][1] = abs(y - self.env.goal_position[1]) / self.env.height
            feature_matrix[n][2] = (
                np.min(
                    [
                        abs(x - self.env.forbidden_states[i][0])
                        for i in range(len(self.env.forbidden_states))
                    ]
                )
                / self.env.width
            )
            feature_matrix[n][3] = (
                np.min(
                    [
                        abs(y - self.env.forbidden_states[i][1])
                        for i in range(len(self.env.forbidden_states))
                    ]
                )
                / self.env.width
            )
            feature_matrix[n][4] = [x, y] == self.env.goal_position
            feature_matrix[n][5] = float(
                (grid.get(x, y) is not None)
                and (grid.get(x, y).type in {"wall", "lava"})
            )

        return feature_matrix

    def _compute_transition_matrix(self) -> tuple[np.ndarray, list[int]]:
        """
        Computes the transition matrix for this environment and find the terminal states

        Returns
        -------
        P : ndarray
            transition matrix
        terminal_states : list
            terminal states of the environment
        """

        # return P, terminal_states
        terminal_states = []

        grid_size = self.env.width * self.env.height
        # add terminal state for computations in which agent infinitely stays once it reached its goal
        self.n_states = grid_size * self.n_orientations + 1

        # Transition matrix of shape (state_space, state_space, action)
        transition_matrix = np.zeros((self.n_states, self.n_states, self.n_actions))

        # Loop over all possible (x, y, orientation) states
        for x in range(1, self.env.width - 1):
            for y in range(1, self.env.height - 1):
                for orientation in range(self.n_orientations):
                    for action in range(self.n_actions):
                        # Reset and set the agent to the desired state
                        self.env.reset(self.seed)
                        self.env.env.agent_pos = np.array((x, y))
                        self.env.env.agent_dir = orientation

                        # Take the action and observe the result
                        _, _, done, _, _ = self.env.step(action)

                        # Get the resulting position and orientation
                        next_pos = self.env.env.agent_pos
                        next_orientation = self.env.env.agent_dir

                        # Current and next state indices
                        current_state = self.env.to_state_index(x, y, orientation)
                        next_state = self.env.to_state_index(
                            next_pos[0], next_pos[1], next_orientation
                        )

                        # Update the transition matrix
                        transition_matrix[current_state, next_state, action] += 1.0

                        if done or (list(next_pos) == self.env.goal_position):
                            terminal_states.append(next_state)

        #all terminal states can only reach the absorbing state
        for s in set(terminal_states):
            transition_matrix[s,:,:] = 0.0
            transition_matrix[s, self.n_states-1, :] = 1.0

        #agent cannot leave absorbing state
        transition_matrix[self.n_states-1,self.n_states-1,:] = 1.0

        return transition_matrix, terminal_states

    def _get_initial_distribution(self) -> np.ndarray:
        """
        computes initial state distribution

        Returns
        -------
        initial_dist : ndarray
        """
        initial_dist = np.zeros(self.n_states)

        self.init_state = self.env.to_state_index(1, 1, 0)
        initial_dist[self.init_state] = 1.0

        return initial_dist

    def _compute_true_reward(self) -> np.ndarray:
        """
        Compute the true reward of the environment given the true reward parameters

        Returns
        -------
        reward : ndarray
            true reward for each state
        """
        reward = [
            (self.theta_reward.dot(self.feature_matrix[i, :])).dot(
                self.feature_matrix[i, :]
            )
            for i in range(self.feature_matrix.shape[0])
        ]

        return np.array(reward)


class CrossingCoordStateWrapper(gym.ObservationWrapper):
    """
    Gym wrapper to define a custom observation state and reward function

    Parameters
    ----------
    env : Env
        environment for which the custom functions are wanted
    seed : int
        seed to fix randomization
    """

    def __init__(self, env: gym.Env, seed:int = None):
        super().__init__(env)
        self.width = env.width
        self.height = env.height
        self.n_orientations = 4  # possible orientations (0-3)
        self.n_actions = 3  # restrict actions to actually used ones in the environment
        self.seed = seed
        self.env.reset(seed = self.seed)
        self.grid = env.grid
        self.forbidden_states = self.compute_forbidden_states()
        self.goal_position = [self.width - 2, self.height - 2]

        # Adjust the observation space to include x, y, and state index
        coord_low = np.array([0, 0])
        coord_high = np.array([self.width - 1, self.height - 1])

        # Assume the original observation space is a dictionary
        self.observation_space = gym.spaces.Dict(
            {
                "original_obs": env.observation_space,
                "coordinates": gym.spaces.Box(
                    low=coord_low, high=coord_high, dtype=np.int32
                ),
                "direction": gym.spaces.Discrete(self.n_orientations),
                "state_index": gym.spaces.Discrete(
                    self.width * self.height * self.n_orientations
                ),
            }
        )

        self.action_space = gym.spaces.Discrete(self.n_actions)

    def to_state_index(self, x: int, y: int, orientation: int) -> int:
        """
        Converts (x, y, orientation) to a unique state index

        Parameters
        ----------
        x : int
            x - coordinate
        y : int
            y - coordinate
        orientation : int
            orientation of the agent

        Returns
        -------
        state_index : int
            state enumeration
        """
        return (y * self.width + x) * self.n_orientations + orientation

    def from_state_index(self, state_index: int) -> tuple[int, int, int]:
        """
        Recomputation of the coordinates and orientation based on state index

        Parameters
        ----------
        state_index : int
            state enumeration

        Returns
        -------
        x : int
            x - coordinate
        y : int
            y - coordinate
        orientation : int
            orientation of the agent
        """
        orientation = state_index % self.n_orientations
        xy_index = state_index // self.n_orientations
        y = xy_index // self.width
        x = xy_index % self.width
        return x, y, orientation

    def compute_forbidden_states(self) -> list[list[int]]:
        """
        Looks up the coordinates of the forbidden (lava) states in order to use them for feature computations

        Returns
        -------
        forbidden_states : list[list[int]]
        """
        forbidden_states = []
        for x in range(1, self.width - 1):
            for y in range(1, self.height - 1):
                if (self.grid.get(x, y) is not None) and (
                    self.grid.get(x, y).type == "lava"
                ):
                    forbidden_states.append([x, y])

        return forbidden_states

    def observation(self, obs) -> dict[any]:
        """
        Custom observation

        Parameters
        ----------
        obs
            original observation

        Returns
        -------
        extended_obs : dict
            new observation
        """
        # Get the agent's current position and orientation
        x, y = self.env.agent_pos
        orientation = self.env.agent_dir
        state_index = self.to_state_index(x, y, orientation)

        # Construct the new observation
        extended_obs = {
            "original_obs": obs,
            "coordinates": np.array([x, y], dtype=np.int32),
            "orientation": orientation,
            "state_index": state_index,
            "goal": [x, y] == self.goal_position,
        }

        return extended_obs

    def step(self, action: int) -> tuple[int, float, bool, bool, any]:
        """
        Overwritten step function to change the default reward function

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

        obs, reward, done, truncated, info = self.env.step(action)
        obs = self.observation(obs)

        state_coordinates = obs.get("coordinates")
        state_index = obs.get("state_index")

        # Implement your custom reward logic
        if state_coordinates is not None and state_index is not None:
            diff_goal_x = abs(state_coordinates[0] - self.goal_position[0]) / self.width
            diff_goal_y = abs(state_coordinates[1] - self.goal_position[1]) / self.height
            diff_lava_x = (
                np.min(
                    [
                        abs(state_coordinates[0] - self.forbidden_states[i][0])
                        for i in range(len(self.forbidden_states))
                    ]
                )
                / self.width
            )
            diff_lava_y = (
                np.min(
                    [
                        abs(state_coordinates[1] - self.forbidden_states[i][1])
                        for i in range(len(self.forbidden_states))
                    ]
                )
                / self.height
            )
            reward = -(diff_goal_x**2 + diff_goal_y**2) + 0.5*(diff_lava_x**2 + diff_lava_y**2)
            if done:
                if (diff_goal_x == 0) and (diff_goal_y == 0):
                    reward += 10
                else:
                    reward -= 10

        return state_index, reward, done, truncated, info

    def reset(self, seed: int) -> tuple[int, any]:
        """
        Overwritten reset function

        Parameters
        ----------
        seed : int

        Returns
        -------
        state : int
            state index
        info
            additional information
        """
        state, info = self.env.reset(seed=seed)
        return self.observation(state).get("state_index"), info
