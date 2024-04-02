import numpy as np
import scipy as sp
from controlgym.envs import PDE
from controlgym.envs.utils import ft_matrix, ift_matrix


class SchrodingerEnv(PDE):
    """
    ### Description

    This environment models the Schrodinger equation, For a single nonrelativistic
    particle in a constant potential, the Schrodinger equation is described by:
        i * planck_constant * du/dt
        =  - planck_constant^2 / (2 * particle_mass) * d^2u/dx^2 + potential * u
    where u is the wave function. Here, this equation is solved as two coupled PDEs
    for the real and imaginary parts of the wave function, Re(u) and Im(u). The domain
    is set to [0, domain_length] with periodic boundary conditions.

    ### Action Space, Observation Space, Rewards, Episode Termination

    See parent class `PDE` defined in pde.py for details.

    ### Arguments

    ```
    controlgym.make('schrodinger')
    ```
    optional arguments:
    [n_steps]: the number of discrete-time steps. Default is 100.
    [domain_length]: domain length of PDE. Default is 1.0.
    [integration_time]: numerical integration time step. Default is None.
    [sample_time]: each discrete-time step represents (ts) seconds. Default is 0.005.
    [process_noise_cov]: process noise covariance coefficient. Default is 0.0.
    [sensor_noise_cov]: sensor noise covariance coefficient. Default is 0.25.
    [target_state]: target state. Default is np.zeros(n_state).
    [init_amplitude_mean]: mean of initial amplitude. Default is 1.0.
    [init_amplitude_width]: width of initial amplitude. Default is 0.2.
    [init_spread_mean]: mean of initial spread. Default is 0.05.
    [init_spread_width]: width of initial spread. Default is 0.02.
    [planck_constant]: planck constant. Default is 1.0.
    [particle_mass]: particle mass. Default is 1.0.
    [potential]: potential. Default is 0.0.
    [n_state]: dimension of state vector. Default is 256.
    [n_observation]: dimension of observation vector. Default is 10.
    [n_action]: dimension of control vector. Default is 8.
    [control_sup_width]: control support width. Default is 0.1.
    [Q_weight]: weight of state tracking cost. Default is 1.0.
    [R_weight]: weight of control cost. Default is 1.0.
    [action_limit]: limit of action. Default is None.
    [observation_limit]: limit of observation. Default is None.
    [reward_limit]: limit of reward. Default is None.
    [seed]: random seed. Default is None.
    """

    def __init__(
        self,
        n_steps: int = 500,
        domain_length: float = 1.0,
        integration_time: float = None,  # Use analytical solution to evolve the dynamics
        sample_time: float = 0.001,
        process_noise_cov: float = 0.0,
        sensor_noise_cov: float = 0.25,
        target_state: np.ndarray[float] = None,
        init_amplitude_mean: float = 1.0,
        init_amplitude_width: float = 0.2,
        init_spread_mean: float = 0.05,
        init_spread_width: float = 0.02,
        planck_constant: float = 1.0,
        particle_mass: float = 1.0,
        potential: float = 0.0,
        n_state: int = 256,
        n_observation: int = 10,
        n_action: int = 8,
        control_sup_width: float = 0.1,
        Q_weight: float = 1.0,
        R_weight: float = 1.0,
        action_limit: float = None,
        observation_limit: float = None,
        reward_limit: float = None,
        seed: int = None,
    ):
        self.n_state_half = int(n_state / 2)
        PDE.__init__(
            self,
            id="schrodinger",
            n_steps=n_steps,
            domain_length=domain_length,
            integration_time=integration_time,
            sample_time=sample_time,
            process_noise_cov=process_noise_cov,
            sensor_noise_cov=sensor_noise_cov,
            target_state=target_state,
            n_state=n_state,
            n_observation=n_observation,
            n_action=n_action,
            control_sup_width=control_sup_width,
            Q_weight=Q_weight,
            R_weight=R_weight,
            action_limit=action_limit,
            observation_limit=observation_limit,
            reward_limit=reward_limit,
            seed=seed,
        )

        # set up the grid parameters: since the state x comprises both the reak and
        # imaginary parts of u, the number of grid points is half the dimension of x
        self.domain_coordinates = np.linspace(
            0,
            self.domain_length - (self.domain_length / self.n_state_half),
            self.n_state_half,
        )

        # set up the physical parameters
        self.planck_constant = planck_constant
        self.particle_mass = particle_mass
        self.potential = potential
        
        # compute A and B2 matrices
        self.A = self._compute_A()
        self.B2 = self.control_sup

        # initial state parameters
        self.init_amplitude_mean = init_amplitude_mean
        self.init_amplitude_width = init_amplitude_width
        self.init_spread_mean = init_spread_mean
        self.init_spread_width = init_spread_width
        self.reset()

    def select_init_state(self, init_amplitude=None, init_spread=None):
        """Function to select the initial state of the PDE."""
        if init_amplitude is None:
            random_amplitude = self.rng.uniform(
                -0.5 * self.init_amplitude_width, 0.5 * self.init_amplitude_width
            )
            init_amplitude = self.init_amplitude_mean + random_amplitude
        if init_spread is None:
            random_spread = self.rng.uniform(
                -0.5 * self.init_spread_width, 0.5 * self.init_spread_width
            )
            init_spread = self.init_spread_mean + random_spread
        init_u = init_amplitude * np.cosh(
            1 / init_spread * (self.domain_coordinates - 1 * self.domain_length / 2)
        ) ** (-1)
        init_v = np.zeros(self.n_state_half)
        init_state = np.concatenate((init_u, init_v))
        return init_state

    def step(self, action: np.ndarray[float]):
        """Run one timestep of the environment's dynamics using the agent actions and optional disturbance input.

        When the end of an episode is reached (``terminated or truncated``), it is necessary to call reset() to
        reset this environment's state for the next episode.

        Args:
            action (`ndarray` with shape `(n_action,)): an action provided by the agent to update the environment state.
                ** Dynamics is evolved based on: state_{t+1} = self.A * state_t + self.B2 * action_t

        Returns:
            observation (`ndarray` with shape `(n_observation,)):
                ** observation is generated by: observation = C * state + noise,
                where C is the observation matrix and noise is a Gaussian random vector with zero
                mean and covariance matrix being self.sensor_noise_cov * I
            reward (float): The reward as the negative quadratic H2 cost of the current stage:
                ** reward = - (state_t.T @ self.Q @ state_t + action_t.T @ self.R @ action_t)
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
        # generate the process noise, which is a Gaussian random vector with dimension n_state
        disturbance = self.rng.multivariate_normal(
            np.zeros(self.n_state),
            self.process_noise_cov * np.identity(self.n_state),
        )

        # compute the observation
        observation = self._get_obs()

        # compute the next state
        next_state = self.A @ self.state + self.B2 @ action + disturbance

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

    def _compute_A(self):
        """Private function to compute analytical solution of A matrix for the linear system.

        Args:
            None.

        Returns:
            A (`ndarray` with shape `(n_state, n_state)).
        """
        DFT = np.vstack(
            (
                np.hstack(
                    (
                        ft_matrix(self.n_state_half),
                        np.zeros((self.n_state_half, self.n_state_half)),
                    )
                ),
                np.hstack(
                    (
                        np.zeros((self.n_state_half, self.n_state_half)),
                        ft_matrix(self.n_state_half),
                    )
                ),
            )
        )
        IDFT = np.vstack(
            (
                np.hstack(
                    (
                        ift_matrix(self.n_state_half),
                        np.zeros((self.n_state_half, self.n_state_half)),
                    )
                ),
                np.hstack(
                    (
                        np.zeros((self.n_state_half, self.n_state_half)),
                        ift_matrix(self.n_state_half),
                    )
                ),
            )
        )
        domain_wavenumbers_complex = (
            np.fft.fftfreq(self.n_state_half, self.domain_length / self.n_state_half)
            * 2
            * np.pi
        )  # wavenumbers
        L12 = np.diag(
            self.planck_constant
            / (2 * self.particle_mass)
            * domain_wavenumbers_complex**2
            + self.potential / self.planck_constant
        )
        L21 = np.diag(
            -self.planck_constant
            / (2 * self.particle_mass)
            * domain_wavenumbers_complex**2
            - self.potential / self.planck_constant
        )
        L = np.vstack(
            (
                np.hstack((np.zeros((self.n_state_half, self.n_state_half)), L12)),
                np.hstack((L21, np.zeros((self.n_state_half, self.n_state_half)))),
            )
        )
        A = np.real(IDFT @ sp.linalg.expm(L * self.sample_time) @ DFT)
        return A

    def _compute_control_sup(self):
        """private function to generate the control support matrix for Schrodinger equation
             based on the user-defined n_state, n_action, and control_sup_width parameters.

        Args:
            None.

        Returns:
            control_sup (`ndarray` with shape `(n_state, n_action)): the control_sup matrix
            determines how each individual control input is distributed in the domain, where
            control_sup_width (float within (0, 1)) defines the width of the domain that each
            control input can affect. It is enforced that control_sup_width * n_actions <= 1.
            Each control input is assumed to be evenly distributed in the domain. In contrast
            to the parent class, the control sup for Schrodinger equation only has nonzero values
            in the second half of the domain, see paper for more descriptions.
        """
        control_sup = np.zeros((self.n_state, self.n_action))
        domain_width = self.control_sup_width * self.n_state_half
        domain_spacing = int(self.n_state_half / self.n_action)
        for u_i in range(self.n_action):
            center_idx = int((0.5 + u_i) * domain_spacing + self.n_state_half)
            left_idx = int(center_idx - 0.5 * domain_width)
            right_idx = int(center_idx + 0.5 * domain_width)
            control_sup[left_idx:right_idx, u_i] = 1
        return control_sup

    def get_params_asdict(self):
        """Save the extra environment parameters as a dictionary.

        Args:
            None.

        Returns:
            a dictionary containing the parameters of the pde environment + extra parameters.
        """
        pde_dict = super().get_params_asdict()
        pde_dict.pop("integration_time")
        extra_data = {
            "planck_constant": self.planck_constant,
            "particle_mass": self.particle_mass,
            "potential": self.potential,
            "init_amplitude_mean": self.init_amplitude_mean,
            "init_amplitude_width": self.init_amplitude_width,
            "init_spread_mean": self.init_spread_mean,
            "init_spread_width": self.init_spread_width,
        }
        return {**pde_dict, **extra_data}
