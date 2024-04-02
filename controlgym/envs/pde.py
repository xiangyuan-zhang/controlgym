import numpy as np
import gymnasium
import math


class PDE(gymnasium.Env):
    """
    ### Description

    This environment provides a template to model PDEs, inheriting from gym.Env class.

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
    ['allen_cahn', 'burgers', 'cahn_hilliard', 'convection_diffusion_reaction',
    'fisher', 'ginzburg_landau', 'korteweg_de_vries', 'kuramoto_sivashinsky',
    'schrodinger', 'wave'], construct the PDE class using the following arguments:

    ```
    controlgym.make('env_id')
    ```

    Additional arguments are supported case by case, see each child classes
    for detailed implementations.
    """

    def __init__(
        self,
        id: str,
        n_steps: int,
        domain_length: float,
        integration_time: float,
        sample_time: float,
        process_noise_cov: float,
        sensor_noise_cov: float,
        target_state: np.ndarray[float],
        n_state: int,
        n_observation: int,
        n_action: int,
        control_sup_width: float,
        Q_weight: float,
        R_weight: float,
        action_limit: float,
        observation_limit: float,
        reward_limit: float,
        seed: int,
    ):
        self.id = id
        self.category = "pde"

        self.n_steps = n_steps
        self.integration_time = integration_time
        self.sample_time = sample_time

        # check whether sample_time is a integer multiple of integration_time
        if integration_time is not None:
            assert (
                round(self.sample_time * 1e8) % round(self.integration_time * 1e8) == 0
            ), "sample_time must be a integer multiple of integration_time."

        self.process_noise_cov = process_noise_cov
        self.sensor_noise_cov = sensor_noise_cov

        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)

        self.n_state = n_state
        # check whether the n_state is even
        assert self.n_state % 2 == 0, "n_state must be even."

        self.target_state = np.zeros(self.n_state) if target_state is None else target_state

        self.n_observation = n_observation
        self.n_action = n_action
        self.control_sup_width = control_sup_width

        # control_sup_width * n_actions must be less than or equal to 1
        assert (
            self.control_sup_width * self.n_action <= 1
        ), "control_sup_width * n_actions must be less than or equal to 1."

        # set up the grid parameters
        self.domain_length = domain_length  # domain length
        self.domain_coordinates = np.linspace(
            0, self.domain_length - (self.domain_length / self.n_state), self.n_state
        )
        self.domain_wavenumbers = (
            np.fft.rfftfreq(self.n_state, self.domain_length / self.n_state) * 2 * np.pi
        )

        # compute control support and observation matrix
        self.control_sup = self._compute_control_sup()
        self.C = self._compute_C()

        # set up the weighting matrices
        self.Q = Q_weight * np.identity(self.n_state)
        self.R = R_weight * np.identity(self.n_action)

        # set up the gym.Env attributes
        self.action_limit = np.inf if action_limit is None else action_limit
        self.observation_limit = (
            np.inf if observation_limit is None else observation_limit
        )
        self.reward_limit = np.inf if reward_limit is None else reward_limit

        self.action_space = gymnasium.spaces.Box(
            low=-self.action_limit,
            high=self.action_limit,
            shape=(self.n_action,),
            dtype=float,
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

    def step(self, action: np.ndarray[float]):
        """Run one timestep of the environment's dynamics using the agent actions.

        When the end of an episode is reached (``terminated or truncated``), it is necessary to call reset() to
        reset this environment's state for the next episode.

        Args:
            action (`ndarray` with shape `(n_action,)): an action provided by the agent to update the environment state.

        Returns:
            observation (`ndarray` with shape `(n_observation,)):
                ** observation is generated by: observation = C * state + noise,
                where C is the observation matrix and noise is a Gaussian random vector with zero
                mean and covariance matrix being self.sensor_noise_cov * I
            reward (float): The reward as the negative quadratic cost of the current stage:
                ** reward = - ((state_t - target_state).T @ self.Q @ (state_t - target_state) + action_t.T @ self.R @ action_t + 2)
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

        # sample the process noise, which is a Gaussian random vector with dimension n_state
        disturbance = self.rng.multivariate_normal(
            np.zeros(self.n_state),
            self.process_noise_cov * np.identity(self.n_state),
        )

        # generate the observation
        observation = self._get_obs()

        # evolve the PDE dynamics using numerical integration
        # for Convection-Diffusion-Reaction, Wave, and Schodinger,
        # use the anlytical solution implemented in the child classes

        # for a PDE written in Fourier space in the form d/dt FFT(u) = L*FFT(u) + FFT(N(u))
        # where FFT(u) denotes the Fourier transform of u;
        # FFT_linear_op returns L
        # FFT_nonlinear_op returns a function mapping from FFT(u) to FFT(N(u))
        fourier_linear_op = self._compute_fourier_linear_op()
        fourier_nonlinear_op = self._compute_fourier_nonlinear_op()

        # number of time-stepping steps
        num_integration_steps = int(self.sample_time // self.integration_time)

        # precompute various ETDRK4 scalar quantities
        E = np.exp(self.integration_time * fourier_linear_op)
        E2 = np.exp(self.integration_time * fourier_linear_op / 2)
        M = 32
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
        LR = self.integration_time * fourier_linear_op[:, np.newaxis] + r[np.newaxis, :]
        Q = self.integration_time * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
        f1 = self.integration_time * np.real(
            np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1)
        )
        f2 = self.integration_time * np.real(
            np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1)
        )
        f3 = self.integration_time * np.real(
            np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1)
        )

        # time-stepping loop
        state_fourier = np.fft.rfft(self.state)
        for _ in range(num_integration_steps):
            # catch RuntimeWarning: overflow
            try:
                fourier_nonlinear_RHS = fourier_nonlinear_op(state_fourier, action)
                a = E2 * state_fourier + Q * fourier_nonlinear_RHS
                Na = fourier_nonlinear_op(a, action)
                b = E2 * state_fourier + Q * Na
                Nb = fourier_nonlinear_op(b, action)
                c = E2 * a + Q * (2 * Nb - fourier_nonlinear_RHS)
                Nc = fourier_nonlinear_op(c, action)
                state_fourier = (
                    E * state_fourier
                    + fourier_nonlinear_RHS * f1
                    + 2 * (Na + Nb) * f2
                    + Nc * f3
                )
            except RuntimeWarning:
                print(
                    "Overflow encountered, choose a smaller integration_time or reduce n_state."
                )
                break

        # set the new system state
        next_state = np.fft.irfft(state_fourier) + disturbance

        # compute the reward, which happens before updating the environment state
        # because the reward (might) depends on both the current state and the next state.
        # * In the default reward function, the dependence on the current state is
        # through the self.state attribute, which will not be updated until the next line.
        reward = self.get_reward(action, observation, disturbance, next_state)

        # update the environment
        self.state = next_state

        # terminated if the cost is too large
        self.step_count += 1
        terminated = False if self.step_count < self.n_steps else True
        truncated = (
            False if self.reward_range[0] <= reward <= self.reward_range[1] else True
        )
        info = {"state": self.state}

        # return the observation and stage cost
        return observation, reward, terminated, truncated, info

    def reset(self, seed: int = None, state: np.ndarray[float] = None):
        """Resets the environment to an initial internal state, returning an initial observation and info.

        This method generates a new starting state with some randomness to ensure that the agent explores the
        state space and learns a generalized policy about the environment. This randomness can be controlled
        with the ``seed`` parameter otherwise if the environment already has a random number generator and
        reset() is called with ``seed=None``, the RNG is not reset.

        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG (`np_random`).
            state (optional `ndarray` with shape `(n_state,)): An specific initial state to reset the environment to.

        Returns:
            observation (`ndarray` with shape `(n_observation,)):
                ** observation is generated by: observation = C * state + noise,
                where C is the observation matrix and noise is a Gaussian random vector with zero
                mean and covariance matrix being self.sensor_noise_cov * I
            info (dict): Contains auxillary information. In this case, it contains the state of the system to be utlized
                        for deploying state-feedback controllers.
        """
        # reset the random number generator if there is a new seed provided
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)

        # randomly generate a state (see child classes for individual implementations)
        if state is None:
            state = self.select_init_state()
        else: # check whether the input state is of the right dimension
            assert state.shape == (
                self.n_state,
            ), "Input state has wrong dimension, the correct dimension is: " + str(
                (self.n_state,)
            )
        self.state = state

        # generate the observation
        observation = self._get_obs()
        info = {"state": self.state}
        self.step_count = 0

        return observation, info

    def set_target_state(self, state: np.ndarray[float]):
        """Set the target state of the system

        Args:
            target_state (`ndarray` with shape `(n_state,)`): the target state of the system

        Returns:
            None.
        """
        # check whether the input target state is of the right dimension
        assert state.shape == (
            self.n_state,
        ), "Input target state has wrong dimension, the correct dimension is: " + str(
            (self.n_state,)
        )
        self.target_state = state

    def _get_obs(self):
        """private function to generate the observation for pdes

        Args:
            None.

        Returns:
            observation (`ndarray` with shape `(n_observation,)): observation = C * state + noise,
                where C is the observation matrix and noise is a Gaussian random vector with zero
                mean and covariance matrix being self.sensor_noise_cov * I
        """
        noise = self.rng.multivariate_normal(
            np.zeros(self.n_observation),
            self.sensor_noise_cov * np.identity(self.n_observation),
        )
        observation = self.C @ self.state + noise
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
        reward = - float((self.state - self.target_state).T @ self.Q @ (self.state - self.target_state) 
                         + action.T @ self.R @ action)
        return reward

    def _compute_fourier_linear_op(self):
        """
        template function to be implemented in child classes for nonlinear PDEs
        """
        raise NotImplementedError

    def _compute_fourier_nonlinear_op(self):
        """
        template function to be implemented in child classes for nonlinear PDEs
        """
        raise NotImplementedError

    def _compute_C(self):
        """private function to generate the C matrix based on the user-defined n_state and n_observation parameters

        Args:
            None.

        Returns:
            C (`ndarray` with shape `(n_observation, n_state)): C is a matrix with only one nonzero entry in each row,
                which is 1. The nonzero entry is located at the position corresponding to the sensor location.
                Each sensors are evenly distributed in the domain.
        """
        C = np.zeros((self.n_observation, self.n_state))
        # loop over all sensors and assign entries of C
        for idx in range(self.n_observation):
            i_x = self.n_state // self.n_observation * idx
            C[idx, i_x] = 1  # set the corresponding entry in C to be 1
        return C

    def _compute_control_sup(self):
        """private function to generate the control support matrix based on the user-defined
            n_state, n_action, and control_sup_width parameters.

        Args:
            None.

        Returns:
            control_sup (`ndarray` with shape `(n_state, n_action)): the control_sup matrix
            determines how each individual control input is distributed in the domain, where
            control_sup_width (float within (0, 1)) defines the width of the domain that each
            control input can affect. It is enforced that control_sup_width * n_actions <= 1.
            Each control input is assumed to be evenly distributed in the domain.
        """
        control_sup = np.zeros((self.n_state, self.n_action))
        domain_width = self.control_sup_width * self.n_state
        domain_spacing = int(self.n_state / self.n_action)
        for u_i in range(self.n_action):
            center_idx = int((0.5 + u_i) * domain_spacing)
            left_idx = int(center_idx - 0.5 * domain_width)
            right_idx = int(center_idx + 0.5 * domain_width)
            control_sup[left_idx:right_idx, u_i] = 1
        return control_sup

    def get_params_asdict(self):
        """save the parameters of the environment as a dictionary

        Args:
            None.

        Returns:
            a dictionary containing the parameters of the pde environment
        """
        env_params =  {
            "id": self.id,
            "n_steps": self.n_steps,
            "domain_length": self.domain_length,
            "integration_time": self.integration_time,
            "sample_time": self.sample_time,
            "process_noise_cov": self.process_noise_cov,
            "sensor_noise_cov": self.sensor_noise_cov,
            "n_state": self.n_state,
            "n_observation": self.n_observation,
            "n_action": self.n_action,
            "control_sup_width": self.control_sup_width,
            "action_limit": self.action_limit,
            "observation_limit": self.observation_limit,
            "reward_limit": self.reward_limit,
        }

        if self.seed is not None:
            env_params["seed"] = self.seed

        if self.target_state is not None:
            env_params["target_state"] = self.target_state

        return env_params
