import numpy as np
from controlgym.envs import PDE
from controlgym.envs.utils import ft_matrix, ift_matrix


class ConvectionDiffusionReactionEnv(PDE):
    """
    ### Description

    This environment models the Convection-Diffusion-Reaction equation, described by:
        du/dt =  diffusivity_constant * d^2u/dx^2
             - convective_velocity * du/dx
             + reaction_constant * u

    ### Action Space, Observation Space, Rewards, Episode Termination

    See parent class `PDE` defined in pde.py for details.

    ### Arguments

    ```
    controlgym.make('convection_diffusion_reaction')
    ```
    optional arguments:
    [n_steps]: the number of discrete-time steps. Default is 100.
    [domain_length]: domain length of PDE. Default is 1.0.
    [integration_time]: numerical integration time step. Default is None.
    [sample_time]: each discrete-time step represents (ts) seconds. Default is 0.1.
    [process_noise_cov]: process noise covariance coefficient. Default is 0.0.
    [sensor_noise_cov]: sensor noise covariance coefficient. Default is 0.25.
    [random_init_state_cov]: random initial state covariance coefficient. Default is 0.0.
    [target_state]: target state. Default is np.zeros(n_state).
    [init_state]: initial state. Default is
        np.cosh(10 * (self.domain_coordinates - 1 * self.domain_length / 2)) ** (-1).
    [diffusivity_constant]: diffusivity constant. Default is 0.002.
    [convective_velocity]: convective velocity. Default is 0.1.
    [reaction_constant]: reaction constant. Default is 0.1.
    [n_state]: dimension of state vector. Default is 200.
    [n_observation]: dimension of observation vector. Default is 10.
    [n_action]: dimension of control vector. Default is 8.
    [control_sup_width]: control support width. Default is 0.1.
    [Q_weight]: weight of state tracking cost. Default is 1.0.
    [R_weight]: weight of control cost. Default is 1.0.
    [action_limit]: limit of action. Default is None.
    [observation_limit]: limit of observation. Default is None.
    [reward_limit]: limit of reward. Default is None.
    [seed]: random seed. Default is 0.
    """

    def __init__(
        self,
        n_steps: int = 100,
        domain_length: float = 1.0,
        integration_time: float = None,  # Use analytical solution to evolve the dynamics
        sample_time: float = 0.1,
        process_noise_cov: float = 0.0,
        sensor_noise_cov: float = 0.25,
        random_init_state_cov: float = 0.0,
        target_state: np.ndarray[float] = None,
        init_state: np.ndarray[float] = None,
        diffusivity_constant: float = 0.002,
        convective_velocity: float = 0.1,
        reaction_constant: float = 0.1,
        n_state: int = 200,
        n_observation: int = 10,
        n_action: int = 8,
        control_sup_width: float = 0.1,
        Q_weight: float = 1.0,
        R_weight: float = 1.0,
        action_limit: float = None,
        observation_limit: float = None,
        reward_limit: float = None,
        seed: int = 0,
    ):
        PDE.__init__(
            self,
            id="convection_diffusion_reaction",
            n_steps=n_steps,
            domain_length=domain_length,
            integration_time=integration_time,
            sample_time=sample_time,
            process_noise_cov=process_noise_cov,
            sensor_noise_cov=sensor_noise_cov,
            random_init_state_cov=random_init_state_cov,
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

        # the highest priority is to use the user-provided initial state
        if init_state is not None:
            self.init_state = init_state
        # if the user does not provide an initial state, use the default initial state
        else:
            self.init_state = np.cosh(
                10 * (self.domain_coordinates - 1 * self.domain_length / 2)
            ) ** (-1)
        self.state = self.init_state
        self.diffusivity_constant = diffusivity_constant
        self.convective_velocity = convective_velocity
        self.reaction_constant = reaction_constant

        # compute A, B2, C
        self.A, self.eigen = self._compute_A()
        self.B2 = self._compute_control_sup()
        self.C = self._compute_C()

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
        # check the dimension of the input control
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

        return observation, reward, terminated, truncated, info

    def _compute_A(self):
        """Private function to compute analytical solution of A matrix for the linear system.

        Args:
            None.

        Returns:
            A (`ndarray` with shape `(n_state, n_state)).
        """
        DFT = ft_matrix(self.n_state)  # discrete fourier transform matrix
        IDFT = ift_matrix(self.n_state)  # inverse fourier transform matrix
        domain_wavenumber_complex = (
            np.fft.fftfreq(self.n_state, self.domain_length / self.n_state) * 2 * np.pi
        )  # wavenumbers
        eigen = np.exp(
                    (
                        -self.convective_velocity * 1j * domain_wavenumber_complex
                        - self.diffusivity_constant * domain_wavenumber_complex**2
                        + self.reaction_constant
                    )
                    * self.sample_time
                )
        A = np.real(IDFT @ np.diag(eigen) @ DFT)
        return A, eigen

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
            "diffusivity_constant": self.diffusivity_constant,
            "convective_velocity": self.convective_velocity,
            "reaction_constant": self.reaction_constant,
        }
        return {**pde_dict, **extra_data}
