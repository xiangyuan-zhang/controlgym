import numpy as np
import gymnasium
import importlib.resources as pkg_resources
import scipy.io as sio
from controlgym.envs import linear_control_src
from controlgym.envs.utils import c2d

class LinearControlEnv(gymnasium.Env):
    """
    ### Description

    This environment provides a template to model linear control problems, inheriting from gym.Env class.

    ### Action Space

    The action is a `ndarray` with shape `(n_action,)` which can take continuous values.

    ### Observation Space

    The observation is a `ndarray` with shape `(n_observation,) which can take continuous values.

    ### Rewards

    The default rewards are set to be the negative of the quadratic stage cost (regulation cost)

    ### Episode End

    The episode ends if any one of the following occurs:

    1. Termination: Episode length is greater than self.n_steps
    2. Truncation: reward goes beyond the range of (-self.reward_range, reward_range)

    ### Arguments
    For env_id in the following list:
    ["toy", "ac1", "ac2", "ac3", "ac4", "ac5", "ac6", "ac7", "ac8", "ac9", "ac10",
    "bdt1", "bdt2", "cbm", "cdp", "cm1", "cm2", "cm3", "cm4", "cm5",
    "dis1", "dis2", "dlr", "he1", "he2", "he3", "he4", "he5", "he6", "iss",
    "je1", "je2", "lah", "pas", "psm", "rea", "umv"]

    ```
    controlgym.make('env_id')
    ```

    optional arguments:
    [n_steps]: the number of discrete-time steps. Default is 1000.
    [sample_time]: each discrete-time step represents (ts) seconds. Default is 0.1.
    [noise_cov]: process noise covariance coefficient. Default is 0.1.
    [random_init_state_cov]: random initial state covariance coefficient. Default is 0.1.
    [init_state]: initial state. Default is 
        saved values in .mat file or np.zeros(self.n_state) + self.noise_cov * np.identity(self.n_state).
    [action_limit]: limit of action. Default is None.
    [observation_limit]: limit of observation. Default is None.
    [reward_limit]: limit of reward. Default is None.
    [seed]: random seed. Default is None.
    """
    def __init__(
        self,
        id: str,
        n_steps: int = 1000,
        sample_time: float = 0.1,
        noise_cov: float = 0.1,
        random_init_state_cov: float = 0.1,
        init_state: np.ndarray[float] = None,
        action_limit: float = None,
        observation_limit: float = None,
        reward_limit: float = None,
        seed: int = None,
    ):
        self.n_steps = n_steps
        self.sample_time = sample_time
        self.noise_cov = noise_cov
        self.random_init_state_cov = random_init_state_cov
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)

        # setup the problem settings
        self.id = id
        self.category = "linear"

        with pkg_resources.path(linear_control_src, f"{self.id}.mat") as mat_path:
            assert mat_path.exists(), "Environment id does not exist!"
            # load the environment mat file
            env = sio.loadmat(str(mat_path))

        # compute the discrete-time linear system parameters
        self.A, self.B1, self.B2, self.C1, self.D11, self.D12 = c2d(env["A"],
            env["B1"], env["B2"], env["C1"], env["D11"], env["D12"], self.sample_time)
        self.C = env["C"]
        self.D21 = env["D21"]

        # set the dimension of the system
        self.n_state = self.A.shape[1]
        self.n_disturbance = self.B1.shape[1]
        self.n_action = self.B2.shape[1]
        self.n_observation = self.C.shape[0]

        # set the default weighting matrices of the linear control problem
        self.Q = self.C1.T @ self.C1
        if self.id in ("ac1", "ac6", "ac8"):
            self.R = max(np.linalg.eig(self.D12.T @ self.D12)[0]) * np.identity(
                self.n_action
            )
        else:
            self.R = self.D12.T @ self.D12
        self.S = self.C1.T @ self.D12

        # set up the initial system state
        # the highest priority is to use the user-defined initial state
        # the second priority is to use the initial state defined in the mat file
        # the lowest priority is to use a random initial state
        if init_state is not None:
            self.init_state = init_state
        elif "x_0" in env.keys():
            self.init_state = env["x_0"].reshape(self.n_state)
        else:
            self.init_state = np.zeros(self.n_state)

        # add Gaussian random noise to the initial state with covaraince matrix
        # being self.random_init_state_cov * I
        self.state = self.rng.multivariate_normal(
            self.init_state,
            self.random_init_state_cov * np.identity(self.n_state),
        )
        self.action_limit = np.inf if action_limit is None else action_limit
        self.observation_limit = np.inf if observation_limit is None else observation_limit
        self.reward_limit = np.inf if reward_limit is None else reward_limit

        self.action_space = gymnasium.spaces.Box(
            low=-self.action_limit, high=self.action_limit, shape=(self.n_action,), dtype=float
        )
        self.observation_space = gymnasium.spaces.Box(
            low=-self.observation_limit,
            high=self.observation_limit,
            shape=(self.n_observation,),
            dtype=float,
        )
        self.reward_range = (-self.reward_limit, self.reward_limit)

        self.step_count = 0
        self.state_traj = None

    def step(self, action: np.ndarray[float], disturbance: np.ndarray[float] = None):
        """Run one timestep of the environment's dynamics using the agent actions and optional disturbance input.

        When the end of an episode is reached (``terminated or truncated``), it is necessary to call reset() to
        reset this environment's state for the next episode.

        Args:
            action (`ndarray` with shape `(n_action,)): an action provided by the agent to update the environment state.
            disturbance (optional `ndarray` with shape `(n_disturbance,))
                            : an disturbance provided by the agent to update the environment state.
                ** Dynamics is evolved based on: state_{t+1} = self.A * state_t + self.B1 * disturbance + self.B2 * action_t

        Returns:
            observation (`ndarray` with shape `(n_observation,)): 
                ** Observation is obtained based on: observation = self.C * state_t + self.D21 * disturbance
            reward (float): The reward as the negative quadratic H2 cost of the current stage:
                ** reward = - ||self.C1 @ self.state + self.D11 @ disturbance + self.D12 @ action||_2^2
                **        = - (state_t.T @ self.Q @ state_t + action_t.T @ self.R @ action_t + 2 * state_t.T @ self.S @ action_t)
            terminated (bool): Whether the agent reach the maximum length of the episode (defined in self.n_Steps).
                                If true, the user needs to call reset().
            truncated (bool): Whether the reward goes out of bound. If true, the user needs to call reset().
            info (dict): Contains auxillary information. In this case, it contains the state of the system to be utlized
                        for deploying state-feedback controllers. 
        """
        # check whether the input control is of the right dimension
        assert action.shape == (
            self.n_action,
        ), "Input control has wrong dimension, the correct dimension is: " + str(
            (self.n_action,)
        )

        if disturbance is None:
            # sample the process noise, which is a Gaussian random vector with dimension n_disturbance
            disturbance = self.rng.multivariate_normal(
                np.zeros(self.n_disturbance),
                self.noise_cov * np.identity(self.n_disturbance),
            )
        else:
            assert disturbance.shape == (self.n_disturbance,), (
                "Input disturbance has wrong dimension, the correct dimension is: "
                + str((self.n_disturbance,))
            )

        # generate the observation
        observation = self._get_obs(disturbance)
        output = self._get_output(action, disturbance)

        # step the system dynamics forward for one discrete step
        next_state = self.A @ self.state + self.B1 @ disturbance + self.B2 @ action

        # compute the reward, which happens before updating the environment state
        # because the reward (might) depends on both the current state and the next state.
        # * In the default reward function, the dependence on the current state is
        # through the self.state attribute, which will not be updated until the next line.
        reward = self.get_reward(action, observation, disturbance, next_state)

        # update the environment
        self.state = next_state

        # terminated if the maximum episode length has been reached
        self.step_count += 1
        terminated = False if self.step_count < self.n_steps else True
        truncated = (
            False if self.reward_range[0] <= reward <= self.reward_range[1] else True
        )
        info = {"state": self.state, "output": output}

        # return the observation and stage cost
        return observation, reward, terminated, truncated, info

    def reset(self, seed: int = None, state: np.ndarray[float] = None):
        """Resets the environment to an initial internal state, returning an initial observation and info.

        This method generates a new starting state often with some randomness to ensure that the agent explores the
        state space and learns a generalized policy about the environment. This randomness can be controlled
        with the ``seed`` parameter otherwise if the environment already has a random number generator and
        reset() is called with ``seed=None``, the RNG is not reset.

        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG (`np_random`).
            state (optional `ndarray` with shape `(n_state,)): An specific initial state to reset the environment to.

        Returns:
            observation (`ndarray` with shape `(n_observation,)): 
                ** Observation is obtained based on: observation = self.C * state_t + self.D21 * disturbance
            info (dict): Contains auxillary information. In this case, it contains the state of the system to be utlized
                        for deploying state-feedback controllers. 
        """
        # reset the random number generator if there is a new seed provided
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)

        # reset the system to a user-defined initial state if there is one
        if state is not None:
            assert state.shape == (
                self.n_state,
            ), "Input state has wrong dimension, the correct dimension is: " + str(
                (self.n_state,)
            )
            self.init_state = state

        # add Gaussian random noise to the initial state with covaraince matrix
        # being self.random_init_state_cov * I
        self.state = self.rng.multivariate_normal(
            self.init_state,
            self.random_init_state_cov * np.identity(self.n_state),
        )

        w = self.rng.multivariate_normal(
            np.zeros(self.n_disturbance),
            self.noise_cov * np.identity(self.n_disturbance),
        )
        # generate the observation
        observation = self._get_obs(w)
        info = {"state": self.state}

        self.step_count = 0
        # return the observation
        return observation, info

    def _get_output(self, action: np.ndarray[float], disturbance: np.ndarray[float]):
        """private function to generate the output

        Args:
            action (`ndarray` with shape `(n_action,)): action provided by the agent to update the environment state.
            disturbance (`ndarray` with shape `(n_disturbance,)): either stochastic or deterministic disturbance input.

        Returns:
            output (`ndarray`): output = C1 * state + D11 * disturbance + D12 * action
        """
        output = self.C1 @ self.state + self.D11 @ disturbance + self.D12 @ action
        return output

    def _get_obs(self, disturbance: np.ndarray[float]):
        """private function to generate the observation for linear systems

        Args:
            disturbance (`ndarray` with shape `(n_disturbance,)): either stochastic or deterministic disturbance input.

        Returns:
            observation (`ndarray` with shape `(n_observation,)): observation = C * state + D21 * disturbance
        """
        observation = self.C @ self.state + self.D21 @ disturbance
        return observation

    def get_reward(self, action: np.ndarray[float], observation: np.ndarray[float], disturbance: np.ndarray[float], 
                   next_state: np.ndarray[float]):
        """ function to generate the reward for the current time step

        Args:
            action (`ndarray` with shape `(n_action,)): action provided by the agent to update the environment state.
            observation (`ndarray` with shape `(n_observation,)): observation = C * state + D21 * disturbance 
            (not used in the default reward function)
            disturbance (`ndarray` with shape `(n_disturbance,)): either stochastic or deterministic disturbance input.
            (not used in the default reward function)
            next_state (`ndarray` with shape `(n_state,)): the next state of the system.
            (not used in the default reward function)

        Returns:
            reward (float): The reward as the negative quadratic H2 cost of the current stage:

        Example of constructing an environment with a custom reward function:
        ```
        def custom_get_reward(self, action, observation, disturbance, next_state):
            return - np.linalg.norm(self.state)**2 - np.linalg.norm(action)**2 
        
        if __name__ == "__main__":
            env = gym.make('env_id', **kwargs)
            env.get_reward = custom_get_reward.__get__(env)
        ```
        """
        reward = - float(self.state.T @ self.Q @ self.state 
                         + action.T @ self.R @ action 
                         + 2 * self.state.T @ self.S @ action)
        return reward

    def get_params_asdict(self):
        """save the parameters of the environment as a dictionary

        Args:
            None.

        Returns:
            a dictionary containing the parameters of the environment
        """
        env_params = {
            "id": self.id,
            "n_steps": self.n_steps,
            "sample_time": self.sample_time,
            "noise_cov": self.noise_cov,
            "random_init_state_cov": self.random_init_state_cov,
            "init_state": self.init_state,
            "action_limit": self.action_limit,
            "observation_limit": self.observation_limit,
            "reward_limit": self.reward_limit,
        }

        # Conditionally add "seed" if it's not None
        if self.seed is not None:
            env_params["seed"] = self.seed

        return env_params
